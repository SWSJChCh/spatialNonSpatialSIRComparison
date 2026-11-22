# Quantifying discrepancies in the final outbreak size between spatial and non-spatial models of pathogen transmission

## Overview
This repository contains Python scripts used to generate data for plots in the publication: *Quantifying discrepancies in the final outbreak size between spatial and non-spatial models of pathogen transmission*.

The code is divided into two folders:

- **synthetic**: Spatial and well-mixed SIR models on synthetic landscapes, in which hosts are placed uniformly at random on a periodic square (Figures 2, 3 and S1).
- **avian**: A hypothetical avian disease outbreak on the landscape of poultry premises in Great Britain (Figure 4).

### Code Author
- Samuel Johnson

### Date
- 22/11/2026

### Requirements
Install these pinned packages:

- numpy==1.22.3
- scipy==1.11.4 (avian only)

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

#### avian/countySirMinimal.py
Compares the spatial and well-mixed final size distributions in one county of Great Britain. A single value of $\beta^{\mathrm{SP}}$ is chosen so that $R_0^{\mathrm{SP}} = 2$ across Great Britain as a whole (Eq. 2), and outbreaks are then simulated on the premises of the county only, neglecting transmission across the county boundary. The script contains functions which:

- Compute the exponential transmission kernel using Euclidean distances, truncated at $d_{\max} = 10$ km.
- Simulate the spatial SIR model, returning either the final outbreak size or the infection times. The simulations use an event-driven construction that gives the same distribution as the direct-method Gillespie algorithm, whose cost per event is too high for counties of thousands of premises.
- Compute the exact final size distribution of the finite-population well-mixed SIR model (Eqs. 5–7), holding only two rows of the jump chain in memory.
- Evaluate the exact likelihood of observed infection times with hidden removals (Text S1) on a grid of $R_0^{\mathrm{WM}}$ values, using a recursion that costs $O(K^2)$ for $K$ observed infections and does not overflow for large outbreaks.
- Compute 1-Wasserstein distances between final size distributions.

The county is simulated from 100 index premises (every premises in counties with fewer than 100), with $10^5$ outbreaks for the spatial final size distribution and a separate pool of 1,000 outbreaks whose infection times are used to fit the well-mixed model to 1,000 datasets of $n = 64$ outbreaks. Outputs a .npz file (`county_<COUNTY>.npz`) containing $R_{0,c}^{\mathrm{SP}}$ (Eq. 9), both final size distributions at $R_0^{\mathrm{WM}} = R_{0,c}^{\mathrm{SP}}$ and the 1-Wasserstein distance between them, $\Delta_{\mathrm{model}}$, the posterior mode of $R_0^{\mathrm{WM}}$ and the 1-Wasserstein distance for every dataset, the mean of $\Delta_{\mathrm{calibration}}$, the pseudo-true value of $R_0^{\mathrm{WM}}$ and the proportion of kernel weight neglected at the county boundary.

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
- **Figure S1**: `calibratedR0.py` for every combination of R0 ∈ {1.0, 1.2, ..., 6.0} and SIGMA ∈ {0.005, 0.010, ..., 0.100}, with N_OUTBREAKS = 1,000, N_OBS = [64], N_DATASETS = 1,000 and R0_GRID of 119 equally spaced values from 0.1 to 6.0

Each run of `matchedR0.py` performs $10^7$ simulations, so parameter combinations are best run in parallel.

#### Avian
Run the script from a folder containing the premises data (`landscape.npz`; see Data):

```
cd avian
python countySirMinimal.py
```

Parameters are set as constants at the top of the script:

- **COUNTY**: Index of the county: its position in the alphabetical list of the 91 ceremonial counties (int)
- **LANDSCAPE**: File of premises data (str)
- **R0**: Basic reproduction number of the spatial model across Great Britain, $R_0^{\mathrm{SP}}$ (float)
- **SIGMA**: Transmission lengthscale, $\sigma$, in km (float)
- **CUTOFF**: Distance, $d_{\max}$, in km beyond which premises do not interact (float)
- **GAMMA**: Removal rate, $\gamma$ (float)
- **N_INDEX**: Number of index premises (int)
- **N_REFERENCE**: Number of spatial outbreaks, shared equally between the index premises (int)
- **N_OBSERVATION_POOL**: Number of spatial outbreaks in the pool from which datasets are drawn (int)
- **N_OBS**: Number of outbreaks, $n$, in each dataset (int)
- **N_DATASETS**: Number of independent datasets (int)
- **T_OBS**: Time up to which infections are observed; the default observes whole outbreaks (float)
- **R0_GRID**: Grid of $R_0^{\mathrm{WM}}$ values on which the posterior is evaluated under a uniform prior (array of float)
- **SEED**: Random seed (int)

The results in the publication use:

- **Figure 4**: `countySirMinimal.py` for every COUNTY from 0 to 90 except 13 (the City of London, which contains a single premises)

Each run of `countySirMinimal.py` performs $10^5$ simulations of the final outbreak size and 1,000 simulations with infection times, so counties are best run in parallel.

### Data
The avian analysis uses the locations of poultry premises recorded in the Great Britain Poultry Register (April 2026) and the county boundaries in Ordnance Survey Boundary-Line.

- **Great Britain Poultry Register**: held by the Animal and Plant Health Agency (APHA) ([dataset record](https://www.data.gov.uk/dataset/a90e617c-270e-4e9b-991e-60d07d8881f8/gb-poultry-register)). The premises data are confidential and are not included in this repository.
- **Ordnance Survey Boundary-Line** (October 2026), ceremonial counties: available under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) from the [OS Data Hub](https://osdatahub.os.uk/downloads/open/BoundaryLine) ([product page](https://www.ordnancesurvey.co.uk/products/boundary-line)).

`countySirMinimal.py` reads the premises data from a .npz file with two arrays: `points_km`, the British National Grid coordinates (in km) of each premises, located at the mean of the register locations of its County Parish Holding (CPH) number, and `county_index`, the county containing each premises (its position in the alphabetical list of ceremonial counties). Of the 179,070 premises with usable locations, the 25 lying more than 2 km outside every county boundary were removed, and the 7 lying just outside every boundary (by at most 0.6 km) were assigned to the county with the nearest boundary.

Contains OS data © Crown copyright and database right 2026.
