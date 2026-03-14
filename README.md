# Enhancing Earth System Modelling with Artificial Intelligence: Emulators vs Hybrid Models

## Description

In this challenge, students are provided with a reference simulation from an idealized ocean model. The goal is to reproduce this simulation using a deep-learning–based emulator.

> **Note:** For the sake of time, we will focus exclusively on the **emulator approach** in this winter school. Hybrid schemes combining coarse-resolution ocean models with learned correction terms are left as an extension for the interested reader.

An emulator is a neural network trained to mimic the time evolution of the ocean state: given the current ocean state and atmospheric forcings, predict the next N timesteps — replacing the expensive numerical solver entirely.

---

## Recommended Reading

- [Pathak et al. 2021 — FourCastNet](https://arxiv.org/pdf/2110.02085)
- [Beucler et al. 2024 — Hybrid ocean modelling](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024GL114318)
- [Brajard et al. 2022 — Machine learning for ocean modelling](https://linkinghub.elsevier.com/retrieve/pii/S1463500322000890)

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

## Getting Started

### 1. Install dependencies
```bash
pip install torch xarray numpy matplotlib
```

### 2. Explore the data
```python
from utils import plot_ocean_temperature, plot_salinity_velocity, plot_amoc

plot_ocean_temperature("ocean_simulation_data_lite.nc")
plot_salinity_velocity("ocean_simulation_data_lite.nc")
plot_amoc("ocean_simulation_data_lite.nc")
```

### 3. Load and split the dataset
```python
import xarray as xr
from torch.utils.data import DataLoader, Subset
from utils import OceanDataset

ds = xr.open_dataset("ocean_simulation_data_lite.nc")
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

This project is split into two phases:

### 🔬 Phase 1 — Development on Google Colab (Lite Dataset)

All development and prototyping is done on **Google Colab** using the lite dataset `ocean_simulation_data_lite.nc`, which is a reduced version of the full simulation designed to fit within Colab's memory and runtime constraints.

The goals of this phase are:
- Explore the dataset and understand the variables
- Build and debug your emulator architecture
- Validate the training loop and autoregressive rollout
- Run quick experiments on model design choices

> **Tip:** Use `OceanDatasetLazy` if you run into memory issues on Colab — it loads only the timesteps needed for each sample rather than the full dataset at once.

### 🚀 Phase 2 — Training on the Full Dataset (Server)

Once your code is validated on the lite dataset, we will discuss scaling up training to the **full dataset** on a dedicated server.
