# FL-Healthcare

Code for the use-case methodology in *"Integrated Challenges and Open Issues of Federated Learning and Artificial Intelligence in Smart Healthcare Systems"* (Section 5).

The framework simulates a federation of 10 hospitals with the Flower framework. It trains FedAvg, FedProx and a centralized reference on four healthcare tasks under IID and non-IID (Dirichlet) partitions, optionally with DP-SGD (Opacus, Rényi DP accounting) and FP16 model transmission. It then reports seven metrics, statistical tests, communication cost and Figures 3–9.

## Settings (Section 5)

| Setting | Value | Flag |
|---|---|---|
| Clients | 10 | `--clients` |
| Communication rounds | 50 | `--rounds` |
| Local epochs per round | 5 | `--local-epochs` |
| Optimizer, learning rate | Adam, 1 × 10⁻³ | `--lr` |
| Batch size | 32 | `--batch-size` |
| Dropout | 0.3 | `--dropout` |
| Non-IID partition | Dirichlet α = 0.5 over class labels | `--partition dirichlet --alpha` |
| FedProx proximal coefficient | μ = 0.01 | `--mu` |
| DP-SGD | σ = 1.1, C = 1.0, δ = 1 × 10⁻⁵, RDP accounting | `--dp --noise-multiplier --max-grad-norm --delta` |
| ε sensitivity | ε ∈ {0.5, 1.0, 2.0}, σ calibrated per client | `--target-epsilon` |
| Seeds | 42, 123, 456, 789, 1024 | `--seed` |
| Models | 3-layer MLP (tabular/signal); ResNet-18, ImageNet-pretrained (images) | `--hidden`, `--weights` |

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # on a GPU machine install the CUDA build of torch/torchvision first
```

## Data

Put the data under `data/`:

| Task | Source | Files |
|---|---|---|
| Arrhythmia detection | MIT-BIH Arrhythmia Database, Kaggle *ECG Heartbeat Categorization* release (109,446 beats) | `data/mitbih/mitbih_train.csv`, `data/mitbih/mitbih_test.csv` |
| 30-day readmission | MIMIC-III v1.4, PhysioNet (credentialed access and data use agreement required) | Run `python scripts/build_mimic_readmission.py --mimic-dir PATH --out data/mimic/mimic_readmission.csv` |
| Skin-lesion classification | ISIC 2019 training set (25,331 images) | `data/isic2019/ISIC_2019_Training_GroundTruth.csv`, `data/isic2019/ISIC_2019_Training_Input/*.jpg` |
| Radiology (synthetic) | Generated deterministically by the code (18,000 images, 50% positive) | none |

Binary targets: MIT-BIH, normal (N) vs abnormal (S, V, F, Q); MIMIC-III, readmitted within 30 days; ISIC 2019, malignant (MEL, BCC, SCC) vs benign; radiology, opacity present. Dirichlet partitioning uses the fine-grained labels (5 beat classes, 8 lesion classes).

For offline machines, download the torchvision ResNet-18 ImageNet weights (`resnet18-f37072fd.pth`) and pass `--weights PATH`. Each run records the weights it actually used (`weights_source` in `config.json`).

## Running

Single run:
```
python run_experiment.py --dataset mitbih --mode fedavg --partition dirichlet --seed 42 \
    --out outputs/mitbih/fedavg_noniid/seed42
```

Full grid (9 conditions × 4 datasets × 5 seeds = 180 runs; finished runs are skipped, so you can stop and restart):
```
python run_all.py                                   # everything
python run_all.py --datasets mitbih mimic           # tabular tasks only
python run_all.py -- --client-gpus 0.2              # pass options to every run (5 clients per GPU)
```

Analysis (tables, statistics, Figures 3–9, `summary.txt`):
```
python analyze.py --out-root outputs --report report
```

## What each run writes

| File | Content | Used for |
|---|---|---|
| `round_log.csv` | per round: pooled validation accuracy and loss, ε, cumulative bytes | Figs 3, 4, 7 |
| `test_metrics.json` | final global model on the held-out test set: 7 metrics, ε, σ | Table 5, Fig 6, Table 3 |
| `client_metrics.csv` | final global model on each client's local hold-out | Fig 9 |
| `partition.csv` | per-client sample and class counts | Fig 8 |
| `comm.json` | parameters, payload per client per round, FL total, raw-data size | Table 4, Fig 5 |
| `config.json` | every setting, per-client σ and q, software and hardware versions | Section 5.1 |

## Method notes

- **Validation and test data.** 10% of the training pool is held out as a pooled server validation set (per seed). The test split is fixed (`SPLIT_SEED`) and is never used during training. Each client also holds out 10% of its own data for client-level evaluation.
- **Centralized reference.** CL trains on the same pool with the same model and optimizer, logging validation accuracy after every epoch (default: as many epochs as FL rounds).
- **Differential privacy.** Sample-level DP-SGD is applied on every client with Poisson sampling. ε is computed with the Rényi DP accountant over all R × E local epochs, and the maximum over clients is reported. Non-DP runs have no finite ε.
- **Communication.** FL total = rounds × clients × 2 (download + upload) × model payload. The centralized comparison uses the size of the raw training data sent once to a central server.
- **Failures.** If any client fails in any round, the run stops, writes no results and renames its log to `round_log.csv.INVALID`.
- **Statistics.** The seed is the unit of analysis, with n = 5 per condition. Reported tests: Welch two-sample t-test, paired t-test across matched seeds, Cohen's d, 95% CI of the difference, and Holm correction across the seven metrics.

## Tests

`tests/run_smoke_test.sh` creates tiny random files in each dataset format, runs every mode for a few rounds, and runs the analysis. It checks that the code works; its numbers are meaningless.

## Citation of software

Flower (Beutel et al., 2020), Opacus (Yousefpour et al., 2021), PyTorch and torchvision.
