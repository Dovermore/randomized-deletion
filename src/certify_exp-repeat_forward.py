import argparse
import os
from collections import ChainMap
# Adapted from randomizedAblation/utils.py
from typing import Optional, Tuple, Union

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from torchmalware.certification import CertifiedMalConv, RandomPerturbation
from torchmalware.utils import CertifyDataset
from torchmalware.types import IntBinarySample
from utils import load_certified_malconv_ckpt, make_dataset


def avg_hard_forward_smpl(
    repeat_probs: torch.Tensor,
    classifier: CertifiedMalConv,
    normalize: bool = False,
) -> torch.IntTensor:
    """Approximates the expected output of a smoothed classifier using given Monte Carlo samples

    Args:
        repeat_probs: Samples of the classifier's predicted probability under a random perturbation. Must have shape
            (num_samples, num_classes).
        classifier: Classifier used to compute the probabilities in `repeat_probs`.

    Keyword args:
        normalize: Returns class frequencies if False, otherwise returns probabilities.

    Returns:
        Tensor with shape (num_classes,)
    """
    classifier.to(repeat_probs.device)
    classifier.eval()

    num_samples, num_classes = repeat_probs.size()
    with torch.no_grad():
        label_counts = torch.sum(classifier._base_proba_reduce(repeat_probs), dim=0).to(torch.int32)

    # Divide by number of samples to get empirical expectation
    return label_counts / num_samples if normalize else label_counts


def certify_smpl(
    sample: IntBinarySample,
    classifier: CertifiedMalConv,
    perturbation: RandomPerturbation,
    repeat_probs_pred: torch.Tensor,
    repeat_probs_bound: torch.Tensor,
    alpha: float = 0.05,
    abstain: Optional[float] = None,
    return_counts: bool = False,
    **kwargs,
) -> Union[
    Tuple[int, float], Tuple[int, float, Tuple[torch.IntTensor, torch.IntTensor]]
]:
    """Certify a smoothed classifier using given Monte Carlo samples

    Args:
        sample: Test sample to certify.
        classifier: Base classifier.
        perturbation: Random perturbation that is applied to raw inputs before being passed to the base classifier.
        alpha: Significance level.
        repeat_probs_pred: Samples of the classifier's predicted probability under a random perturbation. Must have
            shape (num_samples_pred, num_classes).
        repeat_probs_bound: Samples of the classifier's predicted probability under a random perturbation. Must have
            shape (num_samples_bound, num_classes).

    Keyword args:
        abstain: If specified, the smoothed classifier will abstain from making a prediction if the certified radius
            is less than or equal to this value.
        return_counts: If the count data should be returned. This option is added to help reuse some calculation.
        **kwargs: Keyword arguments passed to `perturbation.certified_radius` method

    Returns:
        A tuple containing the predicted class and the radius of certification. A predicted class of "-1" indicates
        an abstained prediction.
    """

    label_counts_pred = avg_hard_forward_smpl(
        repeat_probs_pred, classifier, normalize=False
    )
    pred, pval = perturbation.predict(sample, label_counts_pred)

    label_counts_bound = avg_hard_forward_smpl(
        repeat_probs_bound, classifier, normalize=False
    )
    radius = perturbation.certified_radius(
        sample, pred, label_counts_bound, alpha=alpha, **kwargs
    )

    if abstain is not None:
        # Abstain from making a prediction if the largest radius of certification is zero
        if radius <= abstain:
            pred = -1

    out = (pred, radius)
    if return_counts:
        out += ((label_counts_pred, label_counts_bound),)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Produce certified results from partitions of repeat forward probabilities")
    parser.add_argument("--repeat-conf")
    parser.add_argument("--certify-conf")
    parser.add_argument("--num-partitions", type=int)
    parser.add_argument("--fp-ratio", type=float, required=False, default=None)
    args = parser.parse_args()

    repeat_conf_path = args.repeat_conf
    certify_conf_path = args.certify_conf
    num_partitions = args.num_partitions
    fp_ratio = args.fp_ratio

    # Load forward data given path and config (and collect them into one file)
    repeat_conf = yaml.load(open(repeat_conf_path, "rb"), Loader=yaml.Loader)
    certify_conf = yaml.load(open(certify_conf_path, "rb"), Loader=yaml.Loader)

    if repeat_conf["ckpt"] != certify_conf["ckpt"]:
        print("repeat_conf and certify_conf has different checkpoint.")
        print("Using the certify_conf ckpt:")
        print(f"\t{certify_conf['ckpt']}")

    repeat_name = repeat_conf["name"]
    repeat_probs_dir = repeat_conf["save_dir"]

    metadata = []
    repeat_probs = []
    for partition in range(num_partitions):
        probs_path = os.path.join(repeat_probs_dir, f"{repeat_name}-{partition}_{num_partitions}.ckpt")
        data = torch.load(probs_path)[-1]
        repeat_probs.append(data["repeat_probs"])
        metadata.append(data["metadata"])

    metadata = dict(ChainMap(*metadata))
    repeat_probs = torch.concat(repeat_probs, dim=1)
    idx_path = sorted([(metadata[path]["idx"], path) for path in metadata], key=lambda x: x[0])

    # Compute prediction and CR
    name = certify_conf["name"]
    ckpt_path = certify_conf["ckpt"]
    save_dir = certify_conf["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}-{fp_ratio}.csv')
    if os.path.exists(save_path):
        print(f"Certify results already exist at: {save_path}. Exit without running certify")
        quit()

    ckpt, model_conf, model, perturbation, transform = load_certified_malconv_ckpt(ckpt_path, fp_ratio=fp_ratio)
    # When doing certification, by default use hardmax
    model.reduce = certify_conf.get('reduce', 'hard')

    perturbation.train(False)
    data = repeat_conf['data']
    if data is None:
        data = model_conf['valid_data']

    dataset = make_dataset(data, transform=transform)
    eval_dataset = CertifyDataset(dataset=dataset)

    kwargs = certify_conf["certify_kwargs"]
    num_samples_pred = kwargs["num_samples_pred"]
    num_samples_bound = kwargs["num_samples_bound"]
    alpha = kwargs["alpha"]
    abstain = kwargs.get("abstain", None)
    assert num_samples_pred + num_samples_bound <= repeat_probs.size(0), f"Not enough samples ({num_samples_pred} + {num_samples_bound} = {num_samples_pred + num_samples_bound} > {repeat_probs.size(0)})"
    # Estimate class probabilities for the smoothed classifier using Monte Carlo samples. These samples are used
    # solely to estimate predictions.
    # Shuffle the probs with a fixed seed
    torch.manual_seed(42)
    repeat_probs = repeat_probs[torch.randperm(repeat_probs.size(0))]
    repeat_probs_pred = repeat_probs[:num_samples_pred]
    repeat_probs_bound = repeat_probs[num_samples_pred: (num_samples_bound+num_samples_pred)]

    # Make 0-th dim index instances in the dataset
    repeat_probs_pred = torch.transpose(repeat_probs_pred, 0, 1)
    repeat_probs_bound = torch.transpose(repeat_probs_bound, 0, 1)

    dataset_size, num_samples, num_classes = repeat_probs_bound.size()

    # Store some metainfo about the dataset
    labels = []
    file_sizes = []
    binary_paths = []

    preds = torch.empty(dataset_size, dtype=torch.int64, device=repeat_probs_bound.device)
    radii = torch.empty(dataset_size, dtype=float, device=repeat_probs_bound.device)
    label_counts_pred = torch.empty((dataset_size, num_classes), dtype=float, device=repeat_probs_bound.device)
    label_counts_bound = torch.empty((dataset_size, num_classes), dtype=float, device=repeat_probs_bound.device)

    for i, path in (pbar := tqdm(idx_path)):
        pbar.set_description("Compute certified radius")
        # COMMENT: Metadata cannot be dropped here. Thus we should not apply transform to x (before this, x have to go through __call__ which ensures its BinarySample)
        try:
            # Try do it without loading actual data (to save time)
            preds[i], radii[i], counts = certify_smpl(
                (None, None), model, perturbation, repeat_probs_pred[i], repeat_probs_bound[i], 
                return_counts=True, **kwargs
            )
            label_counts_pred[i] = counts[0]
            label_counts_bound[i] = counts[1]
        except:
            preds[i], radii[i], counts = certify_smpl(
                eval_dataset[i], model, perturbation, repeat_probs_pred[i], repeat_probs_bound[i], 
                return_counts=True, **kwargs
            )
            label_counts_pred[i] = counts[0]
            label_counts_bound[i] = counts[1]
        labels.append(metadata[path]["label"])
        file_sizes.append(metadata[path]["file_size"])
        binary_paths.append(
            os.path.relpath(path, data["path"])
        )

    out = {
        'label': labels,
        'pred': preds,
        'certified_radius': radii,
        'file_size': file_sizes,
        'binary_path': binary_paths,
        **{
            f'label_counts_pred{i}': label_counts_pred[:, i]
            for i in range(label_counts_pred.shape[1])
        },
        **{
            f'label_counts_bound{i}': label_counts_bound[:, i]
            for i in range(label_counts_bound.shape[1])
        },
    }
    out = pd.DataFrame(data=out)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}-{fp_ratio}.csv')
    out.to_csv(save_path)
