import numpy as np

from spatialSirMinimal import buildGraphs, calibrateBeta, runCell

# Spatial versus well-mixed final sizes at matched R0.
R0 = 2.0
SIGMA = 0.1
N_POP = 100
BOX_SIZE = 1.0
GAMMA = 1.0
N_GRAPHS = 10000  
N_REPEATS = 1000
SEED = 1

rng = np.random.default_rng(SEED)
graphs = buildGraphs(N_GRAPHS, N_POP, BOX_SIZE, rng)
beta = calibrateBeta(graphs, SIGMA, R0, GAMMA)
xGrid = np.linspace(0.0, 1.0, 101)
result = runCell(graphs, SIGMA, beta, GAMMA, N_REPEATS, R0, xGrid, rng)
np.savez("matchedR0.npz", r0=R0, sigma=SIGMA, beta=beta, x=xGrid, **result)
