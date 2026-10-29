# PRIME: Prior-Informed Human and AI Learning with a Certified Autonomy Handoff for Cold Start Fraud Detection

This repository contains the official Python implementation and reproduction scripts for **PRIME** (PRior-Informed Model with E-process certification) for cold-start fraud detection.

---

## Overview

In newly launched payment products or emerging markets with zero labelled chargeback history and limited manual review bandwidth, standard supervised models cannot be trained, while anomaly detectors suffer from high false-positive rates and transferred models suffer from distribution shift.

PRIME addresses these challenges through two components:
1. **Prior-Informed Bayesian Logistic Regression**: Combines label-free scores (Isolation Forest anomaly score and log-odds from an established source product model) into the prior mean of a Bayesian logistic regression model. The model ranks transactions from round 0 before any target labels exist, updating its Laplace posterior with each analyst review.
2. **Certified Autonomy Handoff**: Implements an anytime-valid betting supermartingale test on randomly audited transactions. Autonomous blocking is certified only when evidence crosses $1/\delta$, bounding the probability of a false handoff under the null hypothesis by $\delta$.

---

## Repository Structure

```
.
├── code/
│   ├── coldstart.py        # Core Bayesian model (BayesLogit), review loop (run_stream), betting test (EProcess)
│   ├── data.py             # Data loading and label-free preprocessing
│   ├── run_experiments.py  # Main experiment runner (5 tasks, 20 seeds, baselines, sweeps, ablations)
│   ├── run_handoff.py      # Autonomy handoff evaluation and Monte Carlo null verification
│   ├── analyze.py          # Result aggregation into summary metrics
│   ├── summarize.py        # Statistical comparisons and table reproduction
│   └── make_figs.py        # Figure panel data generator
├── requirements.txt
└── README.md
```

---

## Requirements and Installation

Python 3.8 or higher is recommended. Install required packages via:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`

---

## Dataset Setup

The experiments evaluate two publicly available fraud detection benchmarks:

1. **ULB Credit Card Fraud Detection** (`creditcard.csv`):
   - Available on [Hugging Face](https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection)
2. **IEEE-CIS / Vesta Fraud Detection** (`train_transaction.csv`):
   - Available on [Hugging Face](https://huggingface.co/datasets/aliceczr/ieee-fraud-detection)

Download both CSV files and place them into a `data/` directory at the project root:

```
PRIME/
├── data/
│   ├── creditcard.csv
│   └── train_transaction.csv
└── code/
```

---

## Reproducing Results

All scripts should be executed from within the `code/` directory or with `code/` in the path:

```bash
cd code

# 1. Run main experiments across all 5 benchmark tasks (resumable)
# Usage: python run_experiments.py [n_seeds] [n_workers]
python run_experiments.py 20 2

# 2. Run certified autonomy handoff experiments and null check
python run_handoff.py 20 2

# 3. Aggregate results and output metrics
python analyze.py

# 4. Print paper summary tables and Wilcoxon signed-rank tests
python summarize.py

# 5. Export figure panel data (optional)
python make_figs.py
```
