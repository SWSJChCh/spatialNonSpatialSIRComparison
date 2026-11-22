# Quantifying discrepancies in the final outbreak size between spatial and non-spatial models of pathogen transmission

## Overview
This repository contains Python scripts used to generate data for plots in the publication: *Quantifying discrepancies in the final outbreak size between spatial and non-spatial models of pathogen transmission*.

The code is divided into two folders:

- **synthetic**: Spatial and well-mixed SIR models on synthetic landscapes, in which hosts are placed uniformly at random on a periodic square (Figures 2 and 3).
- **avian**: A hypothetical avian disease outbreak on the landscape of poultry premises in Great Britain (Figures 4 and S1).

### Code Author
- Samuel Johnson

### Date
- 22/11/2026

### Requirements
Install these pinned packages:

- numpy==[version]
- [avian requirements to be added]

```
pip install -r requirements.txt
```

### Script Descriptions

#### synthetic/spatialSirMinimal.py
Contains the functions shared by the synthetic-landscape scripts, which:

- Generate landscapes of hosts placed uniformly at random on a periodic square, with one index host per landscape.
- Compute the exponential transmission kernel using periodic (minimum-image) distances, and choose $\beta^{\mathrm{SP}}$ to give a specified value of $R_0^{\mathrm{SP}}$ (Eq. 3).
- Simulate the spatial SIR model, returning either the final outbreak size (embedded jump chain) or the infection times (direct-method Gillespie algorithm).
- Compute the exact final size distribution of the finite-population well-mixed SIR model (Eqs. 5–7).
- Evaluate the exact likelihood of observed infection times with hidden removals (Text S1) on a grid of $R_0^{\mathrm{WM}}$ values.
- Compute exceedance curves, 1-Wasserstein distances between final size distributions, and posterior-predictive final size distributions for nested datasets.

#### synthetic/matchedR0.py
Simulates the spatial SIR model on 10,000 landscapes (1,000 outbreaks per landscape) and compares its final size distribution with the exact well-mixed final size distribution at $R_0^{\mathrm{WM}} = R_0^{\mathrm{SP}}$. Outputs a .npz file (`matchedR0.npz`) containing both final size distributions, their exceedance curves and the 1-Wasserstein distance between them, $\Delta_{\mathrm{model}}$.

#### synthetic/calibratedR0.py
Fits the well-mixed model to the infection times of a pool of 10,000 spatial outbreaks, with removals unobserved, and compares the posterior-predictive final size distribution with the spatial one for $10^4$ nested datasets of $n \in \{1, 2, 4, \ldots, 512\}$ outbreaks. Outputs a .npz file (`calibration.npz`) containing the posterior mode of $R_0^{\mathrm{WM}}$ and the 1-Wasserstein distance for every dataset and value of $n$, together with $\Delta_{\mathrm{model}}$ and the pseudo-true value of $R_0^{\mathrm{WM}}$.

#### avian/[scriptName].py
[Description to be added.]

### Execution

#### Synthetic landscapes
Run the scripts from within the `synthetic` folder, since both import `spatialSirMinimal.py`:

```
cd synthetic
python matchedR0.py
python calibratedR0.py
```

Parameters are set as constants at the top of each script:

- **R0**: Basic reproduction number of the spatial model, $R_0^{\mathrm{SP}}$ (float)
- **SIGMA**: Transmission lengthscale, $\sigma$, in the same units as BOX_SIZE (float)
- **N_POP**: Number of hosts per landscape (int)
- **BOX_SIZE**: Side length of the square landscape, $L$ (float)
- **GAMMA**: Removal rate, $\gamma$ (float)
- **N_GRAPHS**: Number of simulated landscapes (int)
- **N_REPEATS**: Number of simulations per landscape (int)
- **SEED**: Random seed (int)

`calibratedR0.py` takes the following additional parameters:

- **N_OUTBREAKS**: Number of spatial outbreaks in the pool from which datasets are drawn (int)
- **N_OBS**: Numbers of outbreaks, $n$, in each nested dataset (array of int)
- **N_DATASETS**: Number of independent sets of nested datasets (int)
- **T_OBS**: Time up to which infections are observed; the default observes whole outbreaks (float)
- **R0_GRID**: Grid of $R_0^{\mathrm{WM}}$ values on which the posterior is evaluated under a uniform prior (array of float)

The results in the publication use:

- **Figure 2(a)**: `matchedR0.py` for every combination of R0 ∈ {0.6, 0.8, ..., 6.0} and SIGMA ∈ {0.005, 0.010, ..., 0.100} (560 combinations)
- **Figure 2(b)–(d)**: `matchedR0.py` with SIGMA = 0.05 and R0 = 5.6, 2.6 and 1.0
- **Figure 3**: `calibratedR0.py` with R0 = 2.6 and SIGMA = 0.05

Each run of `matchedR0.py` performs $10^7$ simulations, so parameter combinations are best run in parallel.

#### Avian
```
cd avian
python [scriptName].py
```

[Parameters to be added.]

### Data
The avian analysis uses the locations of poultry premises recorded in the Great Britain Poultry Register (April 2026) and the county boundaries in Ordnance Survey Boundary-Line. [Data access details to be added.]

Contains OS data © Crown copyright and database right 2026.
