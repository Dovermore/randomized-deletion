# This script runs training for smoothed and non-smoothed models. The script loads all
# parameters/configuration from a YAML file, which must be passed to the `conf` argument.

import argparse
import os
import time
from collections import Counter
from copy import deepcopy
from typing import Dict, List

import yaml

from torchmalware.transforms.transforms import DropMetadata, ShiftByConstant

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import csv
import os

# filter pytorch UserWarning (the one with non-writable buffer)
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.types import Device

from torchmalware.certification import CertifiedMalConv, perturbations
from torchmalware.transforms import (
    Compose,
    MaskNonInstruction,
    RemovePEHeader,
    ToTensor,
    Trim,
    ZeroPEHeader,
)
from torchmalware.metadata import Metadata
from torchmalware.types import IntBinarySample
from torchmalware.utils import collate_pad, get_gpu_memory, seed_worker, set_seed
from utils import make_dataset


def write_pred(test_pred, test_idx, file_path):
    test_pred = [item for sublist in test_pred for item in sublist]
    with open(file_path, "w", newline="") as csvfile:
        pred_writer = csv.writer(
            csvfile, delimiter=",", quotechar="'", quoting=csv.QUOTE_MINIMAL
        )
        pred_writer.writerows(zip(test_idx, test_pred))


def metadata_to(metadata: Metadata, device: Device = None) -> Metadata:
    for key in metadata.keys():
        if isinstance(metadata[key], torch.Tensor):
            metadata[key] = metadata[key].to(device)
    return metadata


def train_step(
    batch: IntBinarySample,
    model: CertifiedMalConv,
    loss: nn.Module,
    optimizer: optim.Optimizer,
    history: Dict[str, List],
    device: torch.DeviceObjType,
    num_samples: int = 1,
):
    model.reduce = "none"
    (binaries, metadata), targets = batch
    binaries = binaries.to(device)
    metadata = metadata.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    logits = model.forward(
        binaries,
        num_samples=num_samples,
        return_logits=True,
        return_radii=False,
        batch_size=targets.size(0),
        forward_kwargs=dict(metadata=metadata),
    ).reshape(
        num_samples * targets.size(0), model.out_size
    )  # Reshape from 3d to 2d
    # We want to also duplicate the target tensor
    targets = torch.cat([targets] * num_samples, dim=0)
    loss = loss(logits, targets)
    loss.backward()
    preds = logits.argmax(dim=1)
    optimizer.step()
    history["tr_loss"].append(loss.item())
    history["tr_acc"].extend((targets == preds).tolist())
    mem = 0
    if device == "cuda:0":  # NM: not robust if there are multiple GPUs?
        try:
            mem, _ = get_gpu_memory(device)
        except:
            pass
    history["mem"].append(mem)


def valid_step(
    batch: IntBinarySample,
    model: CertifiedMalConv,
    loss: nn.Module,
    history: Dict[str, List],
    device: torch.DeviceObjType,
    num_samples: int = 1,
):
    model.reduce = "soft"
    with torch.no_grad():
        (binaries, metadata), targets = batch
        binaries = binaries.to(device)
        metadata = metadata_to(metadata, device=device)
        targets = targets.to(device)

        logits = model.forward(
            binaries,
            num_samples=num_samples,
            return_logits=True,
            return_radii=False,
            batch_size=targets.size(0),
            forward_kwargs=dict(metadata=metadata),
        )
        loss = loss(logits, targets)
        preds = logits.argmax(dim=1)
        history["val_loss"].append(loss.item())
        history["val_acc"].extend((targets == preds).tolist())
        history["val_pred"].append(preds.tolist())
    model.reduce = "none"


def debug_step(
    batch: IntBinarySample,
    model: CertifiedMalConv,
    loss: nn.Module,
    history: Dict[str, List],
    device: torch.DeviceObjType,
    states: Dict,
    states_path: str,
):
    model.reduce = "none"
    with torch.no_grad():
        (binaries, metadata), targets = batch
        binaries = binaries.to(device)
        metadata = metadata_to(metadata, device=device)
        targets = targets.to(device)

        # Recompute loss to see if performance significantly degraded
        new_logits = model.forward(
            binaries,
            num_samples=1,
            return_logits=True,
            return_radii=False,
            batch_size=targets.size(0),
            metadata=metadata,
        ).reshape(targets.size(0), model.out_size)

        # Use last 10 histories to consider loss divergence
        loss = np.mean(history["tr_loss"][:10])
        new_loss = loss(new_logits, targets).item()
        if new_loss > (2 * loss):
            print(
                f"[step: {total_step}] The loss diverged ({loss} -> {new_loss})"
            )
            # Store the states
            states["loss"] = loss
            states["new_loss"] = new_loss
            os.makedirs(os.path.dirname(states_path), exist_ok=True)
            torch.save(states, states_path)


def scale_mask_grad(mask_ratio, embed_idx=-1):
    scale = (1 - mask_ratio) / mask_ratio

    def f(grad):
        grad[embed_idx] *= scale
        return grad

    return f


def clip_mask_grad(max_norm="max", embed_idx=-1):
    def f(grad):
        with torch.no_grad():
            if embed_idx == -1:
                _embed_idx = grad.size(0) - 1
            if max_norm == "max":
                _max_norm = torch.max(
                    torch.norm(
                        torch.cat([grad[:_embed_idx], grad[_embed_idx + 1 :]]),
                        p=2,
                        dim=1,
                    )
                )
            else:
                _max_norm = max_norm
            norm = torch.norm(grad[_embed_idx])
            scale = 1 if norm <= _max_norm else _max_norm / norm
            # print(_embed_idx, norm, max_norm, scale)
            grad[_embed_idx] *= scale
        return grad

    return f


if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore")

    # Modified from https://github.com/Alexander-H-Liu/MalConv-Pytorch/blob/master/train.py
    # Load config file for experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="The path to configuration file."
    )
    parser.add_argument(
        "--debug", required=False, action="store_true", help="If debug is enabled."
    )
    args = parser.parse_args()

    debug = args.debug
    conf = yaml.load(open(args.config, "r"), Loader=Loader)

    # Set seed
    seed = conf["seed"]
    if seed is None:
        seed = np.random.randint(0, 10000)

    if seed is not None:
        set_seed(seed)

    exp_name = conf["exp_name"] + "_sd_" + str(seed)
    print("Experiment:\t", exp_name)

    log_dir = conf["log_dir"]
    pred_dir = conf["pred_dir"]
    checkpoint_dir = conf["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, exp_name + ".log")
    ckpt_acc_base_path = os.path.join(checkpoint_dir, exp_name)
    pred_path = os.path.join(pred_dir, exp_name + ".pred")

    # Parameters
    if torch.cuda.is_available() and conf["use_gpu"]:
        device = "cuda:0"
    else:
        device = "cpu"
    num_threads = conf["cpu_threads"]
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    num_workers = conf["num_workers"]
    if num_workers is None:
        num_workers = torch.get_num_threads()
    learning_rate = conf["learning_rate"]
    momentum = conf["momentum"]
    weight_decay = conf["weight_decay"]
    max_epoch = conf["max_epoch"]
    num_samples = conf["num_samples"]
    valid_num_samples = conf["valid_num_samples"]
    test_epoch = conf["test_epoch"]
    batch_size = conf["batch_size"]
    data_size = conf["data_size"]
    out_size = conf["out_size"]
    window_size = conf["window_size"]
    scale_grad_by_freq = conf["scale_grad_by_freq"]
    channels = conf["channels"]
    embed_size = conf["embed_size"]
    display_step = conf["display_step"]

    train_sample_size = conf["train_sample_size"]
    max_early_stop = conf["max_early_stop"]

    embed_num = 256

    # Perturbation parameters
    perturbation = perturbations[conf["perturbation"]](
        *conf["perturbation_args"], **conf["perturbation_kwargs"]
    )
    embed_num += perturbation.extra_dim()
    if conf["non_instruction_mask"] is not None:
        embed_num = max(embed_num, conf["non_instruction_mask"] + 1)
    # Padding
    embed_num += 1

    transform = [
        DropMetadata(["binary_path", "exe_section", "header_size"]),
        Trim(length=data_size),
    ]
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
    print("Transforms are:", transform)

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
        reduce="none",
    ).train()

    # Scale/clip gradient of masked byte
    if "masking" in conf["perturbation"].lower():
        h = model.embed_1.weight.register_hook(clip_mask_grad(0.5))
    train_data = conf["train_data"]
    train_dataset = make_dataset(train_data, transform)
    valid_data = conf["valid_data"]
    valid_dataset = make_dataset(valid_data, transform)
    valid_idx = [path for path, cls in valid_dataset.samples]

    print("Training Set:")
    print("\tTotal", len(train_dataset), "files")
    counter = Counter(train_dataset.targets)
    print("\tMalware Count :", counter[1])
    print("\tGoodware Count:", counter[0])
    if train_sample_size:
        print("\t\t Train sample size:", train_sample_size)

    print("Validation Set:")
    print("\tTotal", len(valid_dataset), "files")
    counter = Counter(valid_dataset.targets)
    print("\tMalware Count :", counter[1])
    print("\tGoodware Count:", counter[0])

    if train_sample_size:
        new_size = min(train_sample_size, len(train_dataset))
        train_dataset, _ = random_split(
            train_dataset, [new_size, len(train_dataset) - new_size]
        )

    pin_memory = False
    non_blocking = pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate_pad,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate_pad,
        pin_memory=pin_memory,
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [{"params": model.parameters()}],
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )

    model = model.to(device)
    ce_loss = ce_loss.to(device)

    step_msg = (
        "epoch-{:02d}-step-{:03d}-loss-{:.6f}-acc-{:.4f}-mem-{:5d}({:.1f}%)-time-{:.2f}"
    )
    valid_msg = "epoch-{:02d}-step-{:03d}-tr_loss-{:.6f}-tr_acc-{:.4f}-val_loss-{:.6f}-val_acc-{:.4f}-time-{:.4f}"
    log_msg = "{:02d}, {:03d}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}, {:.2f}"
    history = {
        "tr_loss": [],
        "tr_acc": [],
        "mem": [],
    }
    log = open(log_file_path, "w")
    log.write("epoch,step,tr_loss,tr_acc,val_loss,val_acc,time,max_mem_cuda\n")

    total_mem = 1
    if device == "cuda:0":  # NM: not robust if there are multiple GPUs?
        try:
            _, total_mem = get_gpu_memory(device)
        except:
            pass

    valid_best_acc = 0.0
    total_step = 0

    step_t0 = eval_t0 = time.time()
    training_dt = 0
    early_stop = 0
    for epoch in range(max_epoch):
        step = 0
        if "cuda" in device:
            torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            # Store variables before step
            if debug:
                states = {
                    "batch": batch,
                    "model": deepcopy(model.state_dict()),
                    "optimizer": deepcopy(optimizer.state_dict()),
                }
            train_step(
                batch=batch,
                model=model,
                loss=ce_loss,
                optimizer=optimizer,
                history=history,
                device=device,
                num_samples=num_samples,
            )
            if debug:
                states_path = os.path.join(
                    ckpt_acc_base_path + "_debug",
                    f"states-step_{total_step}.ckpt",
                )
                debug_step(
                    batch=batch,
                    model=model,
                    loss=ce_loss,
                    history=history,
                    device=device,
                    states=states,
                    states_path=states_path
                )

            # Empty cache to potentially help fragmentation?
            # if device == "cuda":
            #    torch.cuda.empty_cache()
            # This way it also measures loading time
            step_t1 = time.time()
            step_dt = step_t1 - step_t0
            step_t0 = step_t1
            training_dt += step_dt

            if (step + 1) % display_step == 0:
                print(
                    step_msg.format(
                        epoch,
                        total_step,
                        np.mean(history["tr_loss"]),
                        np.mean(history["tr_acc"]),
                        int(history["mem"][-1]),
                        history["mem"][-1] / total_mem * 100,
                        step_dt,
                    ),
                    end="\r",
                    flush=True,
                )
            total_step += 1
            step += 1
        max_memory = max(history["mem"])

        # Interupt for validation
        if (epoch + 1) % test_epoch == 0:
            # Testing
            history["val_loss"] = []
            history["val_acc"] = []
            history["val_pred"] = []
            early_stop += 1

            with torch.no_grad():
                for batch in valid_loader:
                    valid_step(
                        batch=batch,
                        model=model,
                        loss=ce_loss,
                        history=history,
                        device=device,
                        num_samples=valid_num_samples,
                    )

            eval_t1 = time.time()
            eval_dt = eval_t1 - eval_t0
            eval_t0 = eval_t1
            print(
                log_msg.format(
                    epoch,
                    total_step,
                    np.mean(history["tr_loss"]),
                    np.mean(history["tr_acc"]),
                    np.mean(history["val_loss"]),
                    np.mean(history["val_acc"]),
                    eval_dt,
                    training_dt,
                    max_memory,
                ),
                file=log,
                flush=True,
            )

            print(
                valid_msg.format(
                    epoch,
                    total_step,
                    np.mean(history["tr_loss"]),
                    np.mean(history["tr_acc"]),
                    np.mean(history["val_loss"]),
                    np.mean(history["val_acc"]),
                    eval_dt,
                )
            )
            model_ckpt = {
                "epoch": epoch,
                "conf": conf,
                "fp_curve": None,
                "state_dict": model.state_dict(),
            }
            ckpt_acc_path = ckpt_acc_base_path + f"-step_{total_step}.ckpt"
            torch.save(model_ckpt, ckpt_acc_path)
            torch.save(model_ckpt, ckpt_acc_base_path+".ckpt")
            print("\tCheckpoint saved at", ckpt_acc_path)
            write_pred(history["val_pred"], valid_idx, pred_path)
            print("\tPrediction saved at", pred_path)
            early_stop = 0

            if early_stop > max_early_stop:
                break

            history["tr_loss"] = []
            history["tr_acc"] = []
            history["mem"] = []

    log.close()
