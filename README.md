# [RS-Del: Robustness Certificates for Sequence Classifiers via Randomized Deletion](https://arxiv.org/abs/2302.01757)

[![Badge](https://img.shields.io/badge/NeurIPS-2023-blue)](https://nips.cc/Conferences/2023) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository hosts the implementation of NeurIPS 2023 paper ["RS-Del: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion"](https://arxiv.org/abs/2302.01757). We implement `RS-Del`, a new mechanism for randomized smoothing of arbitrary black-box sequence classifiers. Input tokens are repeatedly deleted randomly; resulting base classifier inferences are aggregated to form smoothed predictions. `RS-Del`'s predictions are certifiably robust to edit-distance threat models: where an attacker may make insertions, deletions or substitutions when forming adversarial examples. The mechanism and its certifications are suitable for general sequence classifiers, with the paper exploring applications to malware classification specifically.

---

## 📂 Directory Structure

```plaintext
.
├── configs
│   ├── certify-exp                   # Configs for evaluation step
│   ├── models                        # Configs for malware detection models
│   └── repeat-forward-exp            # Configs for sampling step
├── data
│   ├── binaries                      # Executables for training and evaluation
│   └── {test,train,valid}.csv        # CSV files for data partitioning
├── outputs                           # Directory for experimental outputs
├── run_scripts                       # Shell scripts for running experiment steps
└── src                               # Source code directory
    ├── torchmalware                  # Python package with core implementations
    ├── train.py                      # Script for training models
    ├── repeat_forward_exp.py         # Script for sampling perturbed inputs
    ├── fp_curve-repeat_forward.py    # Script for computing FPR curve
    └── certify_exp-repeat_forward.py # Script for computing certified radius
```

---

## 🚀 Reproducing Experiments

### Environment Setup

Before running any experiments, set up the virtual environment using Pip:

```bash
# Clone the repo
git clone https://github.com/dovermore/randomized-deletion
cd randomized-deletion

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 1. **Model Training**

- Train the smoothed model using data augmentation via `src/train.py`.
- Example: See `run_scripts/task1-train.sh`.

```bash
python src/train.py --conf configs/models/$CONFIG_FILE.yaml
```

### 2. **Prediction, Certification & Calibration Sampling**

- Save base model confidence scores via `src/repeat_forward_exp.py`.
- Example: See `run_scripts/task2-repeat_forward.sh`.

```bash
python src/repeat_forward_exp.py --conf configs/repeat-forward-exp/$CONFIG_FILE.yaml
```

### 3. **False-Positive Rate Calibration (Optional)**

- Vary the decision threshold and compute the FPR via `src/fp_curve-repeat_forward.py`.
- Example: See `run_scripts/task3-fp_curve.sh`.

```bash
python src/fp_curve-repeat_forward.py --path model/checkpoint.pth --repeat-conf configs/repeat-forward-exp/$CONFIG_FILE.yaml
```

### 4. **Certification**

- Compute the certified radius via `src/certify_exp-repeat_forward.py`.
- Example: See `run_scripts/task4-certify-repeat_forward.sh`.

```bash
python src/certify_exp-repeat_forward.py --repeat-conf configs/repeat-forward-exp/$CONFIG_FILE.yaml --certify-conf configs/certify-exp/$CONFIG_FILE.yaml
```

---

## 📊 Datasets

Due to licensing constraints, we are unable to provide direct access to the datasets used in our experiments. However, interested readers can obtain or assemble datasets from the following recommended sources:

### Benign

- [VirusTotal](https://www.virustotal.com/)
- Extract system files from [Windows Virtual Machines](https://developer.microsoft.com/en-us/windows/downloads/virtual-machines/)
- Batch install and scrape program files from [Chocolatey](https://chocolatey.org/)

### Malicious

- [VirusTotal](https://www.virustotal.com/)
- [VirusShare](https://virusshare.com/)

To replicate our experiments using your own dataset, please adhere to the guidelines outlined in [`data/README.md`](data/README.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Cite us

```bibtex
@inproceedings{huang2023rsdel,
  author    = {Huang, Zhuoqun and Marchant, Neil and Lucas, Keane and Bauer, Lujo and Ohrimenko, Olya and Rubinstein, Benjamin I. P.},
  title     = {{RS-Del}: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion},
  year      = {2023},
  booktitle = {Advances in Neural Information Processing Systems},
  series    = {NeurIPS},
}
```
