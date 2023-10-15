import os
import torch
from torch.utils.data import Subset

from torchmalware.datasets.utils import save_dataset
from torchmalware.datasets import RawPE
from torchmalware.transforms import AddInsnAddrRanges, AddExeSectionRanges, AddHeaderSize, Compose
from torchmalware.types import ByteBinarySample
from torchmalware.transforms.functional import to_bytes

from typing import Tuple


def writer(elem: Tuple[ByteBinarySample, int], path: str) -> None:
    sample, _ = elem
    binary, metadata = sample
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata_path = path + ".meta"
    with open(path, "wb") as f:
        f.write(to_bytes(binary))
    torch.save(metadata, metadata_path)


def get_path(idx: int) -> str:
    if isinstance(dataset, Subset):
        idx = dataset.indices[idx]
        path = dataset.dataset.samples[idx][0]
    else:
        path = dataset.samples[idx][0]
    return os.path.abspath(path).replace(root_dir, save_dir)

pe_prep = Compose([
    AddInsnAddrRanges(),
    AddExeSectionRanges(),
    AddHeaderSize(),
])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="root directory for raw dataset (conforming to DatasetFolder structure)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="root directory for processed dataset",
    )
    parser.add_argument(
        "--log",
        type=bool,
        required=False,
        default=True,
        help="whether to save redirected stdout/stderr for each instance",
    )
    parser.add_argument(
        "--ext",
        type=str,
        required=False,
        default=None,
        help="comma-separated list of file extensions to read from the raw data directory"
    )
    parser.add_argument(
        "--np",
        type=int,
        required=False,
        default=0,
        help="number of subprocesses to use when reading/saving the data",
    )
    parser.add_argument(
        "--range",
        type=str,
        required=False,
        default=None,
        help="'start,end', index range of data to preprocess",
    )
    args = parser.parse_args()
    args.ext = tuple(ext.strip() for ext in args.ext.split(","))

    root_dir = os.path.abspath(args.root_dir)
    save_dir = os.path.abspath(args.save_dir)
    
    dataset = RawPE(root=root_dir, extensions=args.ext, transform=pe_prep)
    if args.range is not None:
        indices = tuple(range(*[int(i) for i in args.range.split(",")]))
        dataset = Subset(dataset, indices)

    saved_paths = save_dataset(dataset, get_path, num_workers = args.np, writer=writer, log=args.log)
