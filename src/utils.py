import numpy as np
import random

import torch
from pathos.pools import ProcessPool, SerialPool
from tqdm.auto import tqdm
from torchmalware.certification import CertifiedMalConv, perturbations, RandomPerturbation

from torchmalware.datasets import BinaryDataset, CSVDataset
from torchmalware.models import MalConv
from torchmalware.transforms import (
    Compose,
    MaskNonInstruction,
    RemovePEHeader,
    ShiftByConstant,
    ToTensor,
    Trim,
    ZeroPEHeader,
)


def make_dataset(data, transform=None):
    extensions = (".bytes", ".dll", ".exe", "")
    csv_path = data.get("csv", None)
    data_path = data["path"]
    if csv_path is not None:
        dataset = CSVDataset(
            csv_path=csv_path,
            data_path=data_path,
            transform=transform,
        )
    else:
        dataset = BinaryDataset(
            root=data_path, transform=transform, extensions=extensions
        )
    return dataset


def compute_fp_curve(
    repeat_probs: torch.Tensor,
    labels: torch.Tensor,
    perturbation: RandomPerturbation,
    steps: int,
    num_workers: int = torch.get_num_threads(),
    log_thresh: bool = False,
    return_counts: bool = False,
    return_label_counts: bool = False,
):
    """Estimate a false positive rate (FPR) curve for a smoothed model as a function of the 
    decision threshold

    Args:
        repeat_probs: A tensor containing probability scores from the base model for inputs
            perturbed by the smoothing mechanism. The 1st dimension indexes samples from the 
            smoothing mechanism (for a given instance), the 2nd dimension indexes instances 
            in the dataset, and the 3rd dimension indexes classes. 
        labels: A tensor containing the true labels for each instance in the dataset.
        perturbation: Smoothing mechanism used to perturb the inputs.
        steps: Number of decision thresholds at which to evaluate the FPR.
        num_workers: Number of workers to use when dividing the FPR computation for each 
            threshold. Set to zero to disable parallelization.
        log_thresh: If True, chooses the decision threshold grid uniformly in log space, 
            otherwise chooses the decision threshold grid uniformly in the original space.
        return_counts: If True, appends a tuple to the return list containing the number of 
            false positives (for each threshold) and the number of negatives.
        return_label_counts: If True, appends a 2d tensor to the return list containing the 
            number of samples predicted in each class for each instance. 
    
    Returns:
        A list, where the 1st entry is a tensor containing the grid of decision thresholds, 
        and the 2nd entry is a tensor containing the corresponding false positive rates. 
        If `return_counts` or `return_label_counts` are True, additional entries are appended 
        to the list.
    """
    num_samples = labels.size(0)
    num_classes = repeat_probs.size(2)

    # Allocate tensor to count the predicted labels for each instance
    if log_thresh:
        thresholds = torch.flip(torch.logspace(0, 1, steps, base=0.5), (0,)).unsqueeze(1)
    else:
        thresholds = torch.linspace(start=0, end=1, steps=steps).unsqueeze(1)

    # First aggregate label counts (by varying threshold)
    label_counts = torch.zeros((steps, num_samples, num_classes), dtype=int)
    for sample_idx in tqdm(range(num_samples)):
        p_mals = repeat_probs[:, sample_idx, 1]
        # First aggregate label counts (by varying threshold)
        indices = torch.ones(repeat_probs.size(0), dtype=torch.long) * sample_idx
        # Compute predicted label for each sample
        # steps by batch size number of predictions (sample1step1, sample1step2, ... sample2step1, sample2step2 ...)
        preds = (p_mals >= thresholds).type(torch.long).flatten()
        # Make the indices for the first dimension (0 to steps and duplicate by number of examples given by indices)
        size = indices.size(0)
        thresh_indices = torch.arange(steps, dtype=torch.long).repeat_interleave(size)
        # Need to stack indices to have a corresponding size as preds
        indices = indices.repeat(steps)
        # Add to counts
        label_counts.index_put_(
            (thresh_indices, indices, preds),
            torch.ones_like(preds, dtype=int),
            accumulate=True,
        )
    # We now need to run prediction for each of the thresholds (of each example)
    label_counts = label_counts.cpu()
    if num_workers >= 1:
        pool = ProcessPool(num_workers)
    elif num_workers == 0:
        pool = SerialPool()
    else:
        raise ValueError("`num_workers` must be non-negative")

    def compute_sample_preds(sample_idx):
        sample_preds = -torch.ones(steps, dtype=int)
        for step in range(steps):
            counts = label_counts[step, sample_idx]
            pred, pvalue = perturbation.predict(None, counts)
            sample_preds[step] = pred
        return sample_preds

    imap = pool.imap(compute_sample_preds, range(num_samples))
    preds = torch.stack([x for x in tqdm(imap, total=num_samples)], dim=1)

    # Compute FP cases and FP rate
    n_count = torch.sum(labels == 0).cpu()
    _labels = labels.cpu().unsqueeze(0).expand(steps, -1)
    # Get the counts
    fp_counts = torch.sum(((preds == 1) & (_labels == 0)), dim=1)
    fp_ratios = fp_counts / n_count

    out = [thresholds, fp_ratios]
    if return_counts:
        out.append((fp_counts, n_count))
    if return_label_counts:
        out.append(label_counts)
    return out if len(out) > 1 else out[0]


def load_malconv_ckpt(path: str, seed: int = 42, train: bool = False):
    # TODO: should this be _without_ smoothing?
    """Load the checkpoint of a saved MalConv model with smoothing

    Args:
        path (str): Path to the ckpt file
        seed (int, optional): Seed to set after loading (mainly for perturbation). Defaults to 42.
        train (bool, optional): If model/perturbation should be loaded in training mode. Defaults to False.
    
    Returns:
        tuple: tuple of (ckpt, conf, model, transform)
    """
    ckpt = torch.load(path, map_location="cpu")
    conf = ckpt["conf"]
    state_dict = ckpt["state_dict"]

    window_size = conf["window_size"]
    channels = conf["channels"]
    embed_size = conf["embed_size"]
    scale_grad_by_freq = conf.get("scale_grad_by_freq", False)
    embed_num = 256

    perturbation = perturbations[conf["perturbation"]](
        *conf["perturbation_args"], **conf["perturbation_kwargs"]
    )
    embed_num += perturbation.extra_dim()
    if conf["non_instruction_mask"] is not None:
        embed_num = max(embed_num, conf["non_instruction_mask"] + 1)
    embed_num += 1

    transform = [Trim(length=conf["data_size"])]
    if conf["header"] == "remove":
        transform.append(RemovePEHeader())
    elif conf["header"] == "zero":
        transform.append(ZeroPEHeader())
    if conf["non_instruction_mask"] is not None:
        transform.append(MaskNonInstruction(conf["non_instruction_mask"]))
    transform += [
        ToTensor(dtype=torch.int32),
        ShiftByConstant(1),
    ]
    transform = Compose(transform)

    model = MalConv(
        channels=channels,
        window_size=window_size,
        embed_num=embed_num,
        embed_size=embed_size,
        scale_grad_by_freq=scale_grad_by_freq,
    )
    model.load_state_dict(state_dict=state_dict)

    perturbation.train(train)
    model.train(train)

    # Perturbation parameters
    generator = torch.Generator()
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)
    return ckpt, conf, model, perturbation, transform


def load_certified_malconv_ckpt(
    path: str,
    seed: int = 42,
    train: bool = False,
    load_legacy: bool = False,
    fp_ratio: float = None,
):
    """Load the checkpoint of a saved MalConv model with smoothing

    Args:
        path (str): Path to the ckpt file
        seed (int, optional): Seed to set after loading (mainly for perturbation). Defaults to 42.
        train (bool, optional): If model/perturbation should be loaded in training mode. Defaults to False.
    
    Returns:
        tuple: tuple of (ckpt, conf, model, transform)
    """
    ckpt = torch.load(path, map_location="cpu")
    conf = ckpt["conf"]
    state_dict = ckpt["state_dict"]

    out_size = conf.get("out_size", 2)
    window_size = conf["window_size"]
    channels = conf["channels"]
    embed_size = conf["embed_size"]
    scale_grad_by_freq = conf.get("scale_grad_by_freq", False)
    reduce = conf.get("reduce", "none")

    embed_num = 256
    perturbation = perturbations[conf["perturbation"]](
        *conf["perturbation_args"], **conf["perturbation_kwargs"]
    )

    embed_num += perturbation.extra_dim()
    if conf["non_instruction_mask"] is not None:
        embed_num = max(embed_num, conf["non_instruction_mask"] + 1)
    embed_num += 1

    transform = [Trim(length=conf["data_size"])]
    if conf["header"] == "remove":
        transform.append(RemovePEHeader())
    elif conf["header"] == "zero":
        transform.append(ZeroPEHeader())
    if conf["non_instruction_mask"] is not None:
        transform.append(MaskNonInstruction(conf["non_instruction_mask"]))
    transform += [
        ToTensor(dtype=torch.int32),
        ShiftByConstant(1),
    ]
    transform = Compose(transform)

    model = CertifiedMalConv(
        perturbation=perturbation,
        out_size=out_size,
        channels=channels,
        window_size=window_size,
        embed_num=embed_num,
        embed_size=embed_size,
        scale_grad_by_freq=scale_grad_by_freq,
        threshold=None,
        certify_threshold=None,
        reduce="soft",
    ).train()

    # If we explicit specify load legacy format or there is `embed` layer before renaming
    # We make the naming and keys compatible with new model
    if load_legacy or "embed" in list(state_dict.keys()):
        # Converting to list is necessary to prevent mutating iterator
        for k, v in list(state_dict.items()):
            # Fix embed
            if "embed" in k:
                state_dict[k.replace("embed", "embed_1")] = v
                del state_dict[k]
        ## Include new buffers
        state_dict["_reduce"] = torch.tensor(model.REDUCES[reduce])
        for k, v in perturbation.state_dict().items():
            state_dict["perturbation." + k] = v
    model.load_state_dict(state_dict=state_dict)

    # Set the threshold
    if fp_ratio is not None:
        if "fp_curve" not in ckpt or ckpt["fp_curve"] is None:
            raise Exception("fp_ratio specified but fp_curve is not computed yet.")
        thresholds, fp_ratios = ckpt["fp_curve"]
        model.threshold = torch.tensor(thresholds[fp_ratios <= fp_ratio][0].item())
    # Set certify threshold
    certify_threshold = conf["perturbation_kwargs"].get("thresh", None)
    if certify_threshold is not None:
        model.certify_threshold = torch.tensor(certify_threshold)

    model.train(train)
    # Perturbation parameters
    generator = torch.Generator()
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)
    return ckpt, conf, model, perturbation, transform
