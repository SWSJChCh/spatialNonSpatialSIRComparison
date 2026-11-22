import numpy as np

from spatialSirMinimal import (
    buildGraphs,
    calibrateBeta,
    evaluateNestedSets,
    exactWellMixedFinalSizePmf,
    histogramWasserstein1,
    infectionLogLikelihoodGrid,
    simulateGraphHistograms,
    simulateObservationPool,
)

# Calibrate the well-mixed model to n spatial outbreaks, then compare its
# posterior predictive final sizes with the spatial ones.
R0 = 2.0
SIGMA = 0.1  
N_POP = 100
BOX_SIZE = 1.0
GAMMA = 1.0
N_GRAPHS = 10000  
N_REPEATS = 1000  
N_OUTBREAKS = 10000  
N_OBS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])  
N_DATASETS = 10000  
T_OBS = 1000.0  
R0_GRID = np.linspace(0.1, 6.0, 1181)  
SEED = 1

rng = np.random.default_rng(SEED)
graphs = buildGraphs(N_GRAPHS, N_POP, BOX_SIZE, rng)
beta = calibrateBeta(graphs, SIGMA, R0, GAMMA)
spatialHistogram = simulateGraphHistograms(
    graphs, SIGMA, beta, GAMMA, N_REPEATS, rng
).sum(axis=0)
outbreaks = simulateObservationPool(graphs, SIGMA, beta, GAMMA, T_OBS, N_OUTBREAKS, rng)
logLikelihoods = np.array(
    [infectionLogLikelihoodGrid(t, T_OBS, R0_GRID, GAMMA, N_POP) for t in outbreaks]
)
wellMixedPmfs = np.array([exactWellMixedFinalSizePmf(N_POP, r0) for r0 in R0_GRID])

# Without calibration, and in the limit of many outbreaks (the pseudo-true R0).
w1Matched = histogramWasserstein1(
    spatialHistogram, exactWellMixedFinalSizePmf(N_POP, R0)
)
pseudoTrue = logLikelihoods.mean(axis=0).argmax()
w1PseudoTrue = histogramWasserstein1(spatialHistogram, wellMixedPmfs[pseudoTrue])

# Random datasets of the largest n, with those for smaller n nested within them.
w1 = np.empty((N_DATASETS, len(N_OBS)))
modeR0 = np.empty_like(w1)
for dataset in range(N_DATASETS):
    indices = rng.choice(N_OUTBREAKS, size=N_OBS[-1], replace=False)
    w1[dataset], logLikelihood, _ = evaluateNestedSets(
        indices, N_OBS, logLikelihoods, wellMixedPmfs, spatialHistogram
    )
    modeR0[dataset] = R0_GRID[logLikelihood.argmax(axis=1)]
np.savez(
    "calibration.npz",
    r0=R0,
    sigma=SIGMA,
    nObs=N_OBS,
    w1=w1,
    posteriorModeR0=modeR0,
    w1Matched=w1Matched,
    pseudoTrueR0=R0_GRID[pseudoTrue],
    w1PseudoTrue=w1PseudoTrue,
)