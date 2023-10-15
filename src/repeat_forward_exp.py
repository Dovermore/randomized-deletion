import argparse
import os
import time
from datetime import datetime

import torch
import yaml
import numpy as np
from torch.utils.data import Subset

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader, Dumper

# filter pytorch UserWarning (the one with non-writable buffer)
import warnings

from utils import load_certified_malconv_ckpt
from torchmalware.utils import CertifyDataset
from utils import make_dataset
from torchmalware.certification.utils import repeat_forward_ds

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str, required=True, help='The path to configuration file.'
)
parser.add_argument(
    '--num-partitions', type=int, default=1, required=False, help='Total number of partitions this script is divided into.'
)
parser.add_argument(
    '--partition', type=int, default=0, required=False, help='Index of current partition of this script.'
)
args = parser.parse_args()
conf = yaml.load(open(args.config, 'r'), Loader=Loader)

ckpt_path = conf['ckpt']
name = conf['name']
use_gpu = conf['use_gpu']
batch_size = conf.get('batch_size', 48)
verbose = conf.get('verbose', 0)
eval_size = conf.get('eval_size', None)
num_samples = conf['num_samples']
num_workers = conf['num_workers']
save_dir = conf['save_dir']
data = conf['data']

ckpt, model_conf, model, perturbation, transform = load_certified_malconv_ckpt(ckpt_path, load_legacy=False)

device = 'cuda' if use_gpu else 'cpu'
try:
    model.to(device=device)
except RuntimeError as e:
    print('Encountered:', e, 'Switching to CPU training instead.')
    device = 'cpu'
model.eval()

if data is None:
    data = model_conf['valid_data']
dataset = make_dataset(data, transform=transform)

num_partitions = args.num_partitions
partition = args.partition

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

# Get index of data to be computed
if eval_size is None or eval_size >= len(dataset):
    eval_size = len(dataset)
subset_indices = np.sort(np.random.choice(range(len(dataset)), size=eval_size, replace=False))
subset_indices = split(subset_indices, num_partitions)[partition]
eval_dataset = CertifyDataset(Subset(dataset=dataset, indices=subset_indices))

# Store some metainfo about the dataset
metadata = {}
for i, idx in enumerate(subset_indices):
    path, label = dataset.samples[idx]
    file_size = os.path.getsize(path)
    binary_path = os.path.relpath(path, data['path'])
    metadata[path] = {
        'idx': idx,
        'file_size': file_size,
        'label': label,
    }

t0 = time.time()
with torch.no_grad():
    print("verbose: ", verbose)
    repeat_probs = repeat_forward_ds(eval_dataset, model, num_samples, batch_size=batch_size, verbose=verbose, device=device).cpu()
dt = time.time() - t0

out = [{
    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'duration': dt,
    'metadata': metadata,
    'repeat_probs': repeat_probs,
}]
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'{name}-{partition}_{num_partitions}.ckpt')

# Try to load the saved data first if there are any
try:
    out = torch.load(save_path, map_location='cpu') + out
except:
    pass
torch.save(out, save_path)
