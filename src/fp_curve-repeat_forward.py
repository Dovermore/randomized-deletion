# This script estimates the false positive rate curve of the smoothed model as a function of the
# decision threshold. The curve is appended to the model's checkpoint file under the 'fp_curve'
# key. The script relies on the probability score data generated by `repeat_forward_exp.py`.

import argparse
import os
# filter pytorch UserWarning (the one with non-writable buffer)
import warnings
from collections import ChainMap

import torch
import yaml
import pandas as pd

from utils import compute_fp_curve, load_certified_malconv_ckpt


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes FP curve based on repeat_forward results.")
    parser.add_argument(
        '--path', type=str, 
        help='Path to the checkpoint file for which the FP curve needs to be computed.'
    )
    parser.add_argument(
        '--repeat-conf', 
        help='Location of the configuration file that specifies how to perform the repeat-forward process.'
    )
    parser.add_argument(
        "--num-partitions", type=int, default=1, 
        help="Number of partitions the repeat-forward process is divided into."
    )
    parser.add_argument(
        '--dry-run', action="store_true", 
        help='If enabled, outputs a sample FP curve instead of saving it.'
    )
    parser.add_argument(
        "--num-pred", type=int, default=None, 
        help="Number of samples to use for generating the FP curve."
    )
    args = parser.parse_args()

    repeat_conf_path = args.repeat_conf
    num_partitions = args.num_partitions
    repeat_conf = yaml.load(open(repeat_conf_path, "rb"), Loader=yaml.Loader)
    repeat_name = repeat_conf["name"]
    repeat_probs_dir = repeat_conf["save_dir"]
    num_samples_pred = args.num_pred
    metadata = []
    repeat_probs = []
    for partition in range(num_partitions):
        probs_path = os.path.join(repeat_probs_dir, f"{repeat_name}-{partition}_{num_partitions}.ckpt")
        data = torch.load(probs_path)[-1]
        repeat_probs.append(data["repeat_probs"])
        metadata.append(data["metadata"])

    metadata = dict(ChainMap(*metadata))
    metadata = pd.json_normalize([{"path": k, **v} for k, v in metadata.items()]).set_index("idx").sort_index()
    repeat_probs = torch.concat(repeat_probs, dim=1)
    if num_samples_pred is not None:
        repeat_probs = repeat_probs[:num_samples_pred]
    labels = torch.tensor(metadata["label"].tolist())
    ckpt_path = args.path
    exp_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt, conf, model, perturbation, transform = load_certified_malconv_ckpt(ckpt_path)

    num_workers = torch.get_num_threads()
    fp_steps = conf["fp_steps"]
    # Compute fp curve
    fp_curve = compute_fp_curve(
        repeat_probs=repeat_probs,
        labels=labels,
        perturbation=perturbation,
        steps=fp_steps,
        num_workers=0,
    )
    ckpt["fp_curve"] = fp_curve
    if args.dry_run:
        print("Dry run result")
        print("\tThresh:", fp_curve[0], fp_curve[1])
        print("\tRatios:", fp_curve[1])
    else:
        torch.save(ckpt, ckpt_path)
        print("Checkpoint saved at", ckpt_path)

