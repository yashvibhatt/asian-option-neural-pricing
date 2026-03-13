# Accelerating Monte Carlo Pricing of Asian Options using Neural Networks

**IE 517 — Financial Machine Learning | University of Illinois Urbana-Champaign | Group 3**

---

## Overview

Pricing arithmetic Asian options via Monte Carlo simulation is computationally expensive because
it requires simulating thousands of asset price paths for every new parameter set. This project
investigates whether neural networks can serve as fast, accurate surrogate pricers trained once
on synthetically generated option data and then evaluated in microseconds.

We simulate Geometric Brownian Motion (GBM) paths, compute discounted arithmetic-average payoffs
as ground-truth labels, and train five neural network architectures to predict option prices
directly from a simulated path. We also study how models behave under **distribution shift**
(out-of-distribution parameter regimes) and whether **adaptive sampling** of high-error regions
improves generalization.

---

## Models Compared

| Architecture | Description |
|---|---|
| **MLP** | Flattened path → fully connected layers |
| **CNN** | 1D convolution over the price path |
| **LSTM** | Recurrent model over sequential path steps |
| **CNN-LSTM** | CNN feature extraction followed by LSTM |
| **TCN** | Temporal Convolutional Network with dilated causal convolutions |

---

## Key Results

### Test Set Performance (in-distribution)

| Model | MAE | RMSE |
|---|---|---|
| MLP | **0.0256** | **0.0860** |
| CNN | 0.0420 | 0.1060 |
| LSTM | 0.0407 | 0.1065 |
| CNN-LSTM | 0.2003 | 0.3058 |
| TCN | 0.2137 | 0.3065 |

The MLP achieves the best in-distribution accuracy, outperforming all sequence-based models.
Despite being designed for sequential data, LSTM and CNN perform comparably; CNN-LSTM and TCN
overfit under the current data scale.

### Neural Network vs Monte Carlo Benchmarking (CNN pricer)

- Average relative error across parameter space: **0.394%**
- Neural network pricing is orders of magnitude faster than running 50,000-path MC simulations

### Adaptive Sampling

Oversampling high-error regions did not consistently reduce error in those regions, suggesting
that the model's capacity and the size of the overall dataset are the binding constraints —
not the sampling distribution.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── asian_option_pricing_neural_networks.ipynb   ← main notebook (run this)
│
├── reports/
│   ├── IE517-Final-Report.pdf
│   ├── IE517-Group3-Presentation.pdf
│   └── IE517-Research-Proposal.pdf
│
└── archive/
    └── IE517_Final_Project_American_Baseline.ipynb  ← earlier baseline (American options + sklearn)
```

---

## Setup

### Requirements

- Python 3.9+
- PyTorch 2.0+
- No GPU required (CPU training completes in ~5–10 minutes)

### Install dependencies

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda create -n asian-options python=3.10
conda activate asian-options
pip install -r requirements.txt
```

> **Note:** QuantLib is **not** required for the main notebook. All Monte Carlo simulation uses
> a custom NumPy implementation. QuantLib is only used in the archived American option baseline.

---

## How to Run

1. Open `notebooks/asian_option_pricing_neural_networks.ipynb`
2. Run all cells top-to-bottom (**Kernel → Restart & Run All**)
3. The notebook is self-contained — it generates all data, trains all models, and produces all figures
4. Estimated runtime: ~5–10 minutes on CPU

### Execution order

The notebook must be run sequentially from Cell 1. There are no external data files to download.
All synthetic data is generated in Section 2 using the fixed global seed (`SEED = 42`).

---

## Notebook Structure

| Section | Content |
|---|---|
| **0** | Environment setup and imports |
| **1** | GBM path simulation and Asian option payoff function |
| **2** | Parameter sampling and dataset generation (with EDA) |
| **3** | PyTorch Dataset and DataLoader setup |
| **4** | Model architectures: MLP, CNN, LSTM, CNN-LSTM, TCN |
| **5** | Training loop, evaluation metrics, training all models, loss curves |
| **6** | Benchmarking NN pricers vs high-precision Monte Carlo; price + delta estimation |
| **7** | Pricing curves across spot prices for all models |
| **Ext. A** | Parameter sensitivity analysis, adaptive sampling, cross-distribution generalization |

---

## Reports

- [Final Report](reports/IE517-Final-Report.pdf)
- [Presentation Slides](reports/IE517-Group3-Presentation.pdf)
- [Research Proposal](reports/IE517-Research-Proposal.pdf)

---

## Reproducibility

All results are fully reproducible from a fresh run. Random seeds are set globally at startup:

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

No external datasets or pre-trained weights are required.
