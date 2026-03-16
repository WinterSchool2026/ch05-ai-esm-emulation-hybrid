# Enhancing Earth System Modelling with Artificial Intelligence: Emulators vs Hybrid Models

## Description

In this challenge, students are provided with a reference simulation from an idealized ocean model. The goal is to reproduce this simulation using a deep-learning–based emulator.

> **Note:** For the sake of time, we will focus exclusively on the **emulator approach** in this winter school. Hybrid schemes combining coarse-resolution ocean models with learned correction terms are left as an extension for the interested reader.

An emulator is a neural network trained to mimic the time evolution of the ocean state: given the current ocean state and atmospheric forcings, predict the next N timesteps — replacing the expensive numerical solver entirely.

---

## Recommended Reading

- [Dheeshjith et al. 2025 — Hybrid ocean modelling](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024GL114318)
- [Ronneberger et al. 2022 — UNET model](https://arxiv.org/pdf/1505.04597)

---

## Repository Structure
```
.
├── utils.py          # OceanDataset, OceanDatasetLazy, and visualization functions
├── notebook.ipynb    # Main notebook: data exploration, preprocessing, model training
├── ocean_simulation_data_lite.nc        # Training dataset
└── ocean_simulation_generalization.nc   # Generalization dataset (different forcing)
```

---

## Dataset

The dataset is a NetCDF file containing an idealized ocean simulation with the following variables:

| Variable | Dims | Description |
|---|---|---|
| `temp` | (Time, zt, yt, xt) | Potential temperature (°C) |
| `salt` | (Time, zt, yt, xt) | Salinity (psu) |
| `u` | (Time, zt, yt, xt) | Zonal velocity (m/s) |
| `v` | (Time, zt, yt, xt) | Meridional velocity (m/s) |
| `ssh` | (Time, yt, xt) | Sea surface height (m) |
| `surface_taux` | (Time, yt, xt) | Zonal wind stress (N/m²) |
| `surface_tauy` | (Time, yt, xt) | Meridional wind stress (N/m²) |
| `ml_qnet` | (Time, yt, xt) | Net heat flux (W/m²) |
| `ml_qsol` | (Time, yt, xt) | Solar heat flux (W/m²) |

---

## Getting Started on Google Colab

All development is done on **Google Colab** using Google Drive as persistent storage. Follow these steps once before starting.

### Step 1 — Download the dataset to your PC

Click the link below to download the dataset (~5 GB):

📦 [Download ocean_simulation_data_lite.nc](https://drive.google.com/file/d/13cdYCHNpf2Apgf4YGUGhfQmULbSoxAjn/view?usp=sharing)

### Step 2 — Upload the dataset to your Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Create a folder `YOUR_FOLDER` for the project
3. Upload `ocean_simulation_data_lite.nc` into `YOUR_FOLDER`

### Step 3 — Download the GitHub repo into the same Drive folder

1. On this GitHub page click **Code → Download ZIP**
2. Unzip it on your PC
3. Upload the contents (`utils.py`, `notebook.ipynb`, etc.) into `YOUR_FOLDER`

Your Drive folder should now look like:
```
My Drive / YOUR_FOLDER /
├── utils.py
├── notebook.ipynb
└── ocean_simulation_data_lite.nc
```

### Step 4 — Open the notebook from Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Navigate to `YOUR_FOLDER`
3. Double-click `notebook.ipynb` — it opens directly in Colab

### Step 5 — Mount Drive and set your folder path

At the top of the notebook, run:
```python
from google.colab import drive
drive.mount('/content/drive')

FOLDER = "/content/drive/MyDrive/YOUR_FOLDER"

%cd {FOLDER}
```

> **Only `FOLDER` needs to be changed.** On every new Colab session, just re-run this cell — Drive remounts in seconds and all your files are still there.

**Save checkpoints to Drive in training** so they persist across sessions:
```python
import torch
torch.save(model.state_dict(), "checkpoint.pt")
```

---

## Dataset Loading
```python
from torch.utils.data import DataLoader, Subset

dataset = OceanDataset(ds, input_steps=1, target_steps=5, normalize=True)

n_total = len(dataset)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)
n_test  = n_total - n_train - n_val

train_ds = Subset(dataset, range(0, n_train))
val_ds   = Subset(dataset, range(n_train, n_train + n_val))
test_ds  = Subset(dataset, range(n_train + n_val, n_total))

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)
```

> **Memory tip:** If Colab crashes when loading the dataset, switch to `OceanDatasetLazy` which reads from disk per sample and avoids loading everything into RAM:
> ```python
> dataset = OceanDatasetLazy(data_file, input_steps=1, target_steps=5, normalize=True)
> ```
**Note:** You might need to experiment with the number of data points used to compute the statistics for normalisation, as this step still requires the full dataset.


Each batch contains:

| Key | Shape | Description |
|---|---|---|
| `state_3d` | (B, input_steps, 4, zt, yt, xt) | 3D ocean state (temp, salt, u, v) |
| `state_2d` | (B, input_steps, 1, yt, xt) | 2D ocean state (ssh) |
| `forcings` | (B, input_steps+target_steps, 4, yt, xt) | Atmospheric forcings |
| `target_3d` | (B, target_steps, 4, zt, yt, xt) | Target 3D state |
| `target_2d` | (B, target_steps, 1, yt, xt) | Target 2D state |

> **Note on splitting:** The split is strictly sequential — no future timesteps leak into training. Shuffling is only applied within the training set, since each sample is a self-contained (input → target) window.

---

## Evaluation Challenges

Your emulator will be evaluated on four axes:

### ⚡ 1. Speed — How many Simulated Years Per Day (SYPD)?
Train your emulator and measure how many years of ocean simulation it can produce per wall-clock day. Compare this to the reference numerical model.

### 📈 2. Stability & Long-term Variability
Run the emulator autoregressively over long timescales. Use the provided plotting functions to check:
- Climatological SST and SSS
- Zonal mean stratification
- AMOC index time series

### 🌍 3. Generalization to Different Forcing Conditions
Evaluate your trained emulator on `ocean_simulation_generalization.nc`, which contains a simulation run under different atmospheric forcing conditions not seen during training.

### 🎲 4. Ensemble Generation
Propose a methodology to generate an ensemble of trajectories from your emulator (e.g. perturbations of initial conditions, stochastic components) and evaluate the ensemble spread.

---

## Notebook Outline

The main notebook `notebook.ipynb` is structured as follows:

1. **Visualization** — explore the dataset with the provided plot functions
2. **Data processing** — load, normalize, and split the dataset
3. **Model construction** — build a CNN (or any architecture of your choice)
4. **Training / Validation loop** — train the emulator and monitor validation loss
5. **Test set evaluation** — evaluate on held-out data
6. **Autoregressive rollout** — run the emulator autoregressively and assess stability

---

## Workflow

### 🔬 Phase 1 — Development on Google Colab (Lite Dataset)

All development and prototyping is done on **Google Colab** using the lite dataset `ocean_simulation_data_lite.nc`, which is a reduced version of the full simulation designed to fit within Colab's memory and runtime constraints.

The goals of this phase are:
- Explore the dataset and understand the variables
- Build and debug your emulator architecture
- Validate the training loop and autoregressive rollout
- Run quick experiments on model design choices

### 🚀 Phase 2 — Training on the Full Dataset (Server)

Once your code is validated on the lite dataset, we will discuss scaling up training to the **full dataset** on a dedicated server. The full dataset is significantly larger (~160 GB) so `OceanDatasetLazy` will be used instead of `OceanDataset`.
