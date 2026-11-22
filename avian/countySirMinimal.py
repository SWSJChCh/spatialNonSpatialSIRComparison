import heapq

import numpy as np
from scipy.signal import lfilter
from scipy.spatial import cKDTree
from scipy.special import logsumexp

# Spatial versus well-mixed final sizes in one county of Great Britain, with R0
# matched (Delta_model) and calibrated to n outbreaks (Delta_calibration).
# Index of the county: its position in the alphabetical list of the 91 ceremonial
# counties (e.g. 27 = Durham).
COUNTY = 27
# Premises data (confidential, not distributed): points_km, the British National
# Grid coordinates (km) of every premises, and county_index, the county of each.
LANDSCAPE = "landscape.npz"
R0 = 2.0
SIGMA = 0.5
CUTOFF = 10.0
GAMMA = 1.0
N_INDEX = 100
N_REFERENCE = 100000
N_OBSERVATION_POOL = 1000
N_OBS = 64
N_DATASETS = 1000
T_OBS = 1000.0
R0_GRID = np.linspace(0.0, 8.0, 161)
SEED = 20260723

# Neighbours of premises i within the cutoff distance, by increasing distance (ties
# broken by index), with the cumulative kernel exp(-d / sigma) along them. Distances
# are Euclidean.
def kernelRow(tree, points, i, sigma, cutoff):
    neighbours = np.array(tree.query_ball_point(points[i], cutoff), dtype=np.int64)
    neighbours = neighbours[neighbours != i]
    distance = np.sqrt(np.sum((points[neighbours] - points[i]) ** 2, axis=1))
    order = np.lexsort((neighbours, distance))
    cumulativeKernel = np.cumsum(np.exp(-distance[order] / sigma))
    return neighbours[order], cumulativeKernel

# Truncated exponential kernel of a landscape, stored row by row. Pairs of premises
# more than cutoff apart do not interact.
def buildKernel(points, sigma, cutoff):
    tree = cKDTree(points)
    rows = [kernelRow(tree, points, i, sigma, cutoff) for i in range(len(points))]
    cumulativeKernels = [cumulativeKernel for _, cumulativeKernel in rows]
    return {
        "neighbours": [neighbours for neighbours, _ in rows],
        "cumulativeKernels": cumulativeKernels,
        "kernelSums": np.array([row[-1] if len(row) else 0.0 for row in cumulativeKernels]),
    }

# Kernel sum, sum_j exp(-d_ij / sigma), of every premises, without storing the kernel
# (for the whole of Great Britain).
def kernelSums(points, sigma, cutoff):
    tree = cKDTree(points)
    sums = np.zeros(len(points))
    for i in range(len(points)):
        _, cumulativeKernel = kernelRow(tree, points, i, sigma, cutoff)
        if len(cumulativeKernel):
            sums[i] = cumulativeKernel[-1]
    return sums

# Match the mean initial infection pressure over all premises, in units of gamma, to
# r0 (Eq. 2).
def calibrateBeta(kernelSums, r0, gamma):
    beta = r0 * gamma / np.mean(kernelSums)
    return beta

# Targets of nContacts contacts made by one premises, drawn in proportion to the kernel.
def contactTargets(kernel, premises, nContacts, rng):
    neighbours = kernel["neighbours"][premises]
    cumulativeKernel = kernel["cumulativeKernels"][premises]
    draws = rng.random(nContacts) * cumulativeKernel[-1]
    positions = np.searchsorted(cumulativeKernel, draws, side="right")
    # Guard against a draw equal to the row total after rounding.
    return neighbours[np.minimum(positions, len(neighbours) - 1)]

# For the final size distribution, we use the standard event-driven construction of
# the spatial SIR model, since the direct-method Gillespie algorithm (O(N) per event)
# is too slow for counties of thousands of premises. Each infected premises makes
# Poisson(beta * kernelSum * T) contacts during its Exp(gamma) infectious period, T,
# spread over its neighbours in proportion to the kernel, and the premises ever
# infected are those that the contacts reach (in any order).
# Please note that this gives the same distribution as the Gillespie algorithm.
def simulateFinalSize(kernel, beta, gamma, indexPremises, rng):
    infected = {indexPremises}
    unprocessed = [indexPremises]
    while unprocessed:
        premises = unprocessed.pop()
        kernelSum = kernel["kernelSums"][premises]
        if kernelSum <= 0.0:
            continue
        nContacts = rng.poisson(beta / gamma * kernelSum * rng.standard_exponential())
        if nContacts == 0:
            continue
        for target in contactTargets(kernel, premises, nContacts, rng).tolist():
            if target not in infected:
                infected.add(target)
                unprocessed.append(target)
    return len(infected)

# Final-size histograms of nRepeats outbreaks from each index premises (one row each).
def simulateIndexHistograms(kernel, beta, gamma, indexPremises, nRepeats, rng):
    nPop = len(kernel["kernelSums"])
    histograms = np.zeros((len(indexPremises), nPop + 1), dtype=int)
    for position, premises in enumerate(indexPremises):
        for _ in range(nRepeats):
            finalSize = simulateFinalSize(kernel, beta, gamma, int(premises), rng)
            histograms[position, finalSize] += 1
    return histograms

# Infection times (excluding the index premises, infected at time 0) for the
# likelihood with hidden removals. The same construction as simulateFinalSize, with
# each contact at a uniform time in the infectious period, and premises infected in
# time order from an event queue.
def simulateInfectionTimes(kernel, beta, gamma, indexPremises, rng):
    infected = set()
    infectionTimes = []
    queue = [(0.0, indexPremises)]
    while queue:
        time, premises = heapq.heappop(queue)
        if premises in infected:
            continue
        infected.add(premises)
        if premises != indexPremises:
            infectionTimes.append(time)
        kernelSum = kernel["kernelSums"][premises]
        if kernelSum <= 0.0:
            continue
        infectiousPeriod = rng.standard_exponential() / gamma
        nContacts = rng.poisson(beta * kernelSum * infectiousPeriod)
        if nContacts == 0:
            continue
        targets = contactTargets(kernel, premises, nContacts, rng)
        contactTimes = time + rng.random(nContacts) * infectiousPeriod

        # Only the first contact with each target can infect it.
        order = np.lexsort((contactTimes, targets))
        targets, contactTimes = targets[order], contactTimes[order]
        first = np.r_[True, targets[1:] != targets[:-1]]
        for target, contactTime in zip(
            targets[first].tolist(), contactTimes[first].tolist()
        ):
            if target not in infected:
                heapq.heappush(queue, (contactTime, target))

    return np.array(infectionTimes)

# nPerIndex outbreaks from each index premises, observed up to tObs.
def simulateObservationPool(kernel, beta, gamma, indexPremises, nPerIndex, tObs, rng):
    observations = []
    for premises in indexPremises:
        for _ in range(nPerIndex):
            infectionTimes = simulateInfectionTimes(
                kernel, beta, gamma, int(premises), rng
            )
            observations.append(infectionTimes[infectionTimes <= tObs])
    return observations

# Propagate probability through the exact finite-population well-mixed jump chain
# (Eqs. 5-7), one number of susceptibles, s, at a time. Within each s, the chain of
# removals, pi(s, i) = entering(s, i) + (1 - p_s) pi(s, i + 1), is a first-order
# linear recursion that lfilter evaluates in one call, so only two rows are held in
# memory (counties have up to 7,967 premises). Rates are in units of gamma, which
# does not affect the final size.
def exactWellMixedFinalSizePmf(nPop, r0):
    pairInfectionRate = r0 / (nPop - 1)
    finalSizePmf = np.zeros(nPop + 1)
    # Probability of entering (s, i) by an infection (or at the start).
    entering = np.zeros(nPop + 2)
    entering[1] = 1.0
    for susceptible in range(nPop - 1, -1, -1):
        maxInfected = nPop - susceptible
        relativeInfectionRate = pairInfectionRate * susceptible
        infectionProbability = relativeInfectionRate / (relativeInfectionRate + 1.0)
        recoveryProbability = 1.0 - infectionProbability
        visited = lfilter(
            [1.0], [1.0, -recoveryProbability], entering[maxInfected::-1]
        )[::-1]
        finalSizePmf[maxInfected] = visited[0]
        entering = np.zeros(nPop + 2)
        entering[2 : maxInfected + 2] = visited[1 : maxInfected + 1] * infectionProbability
    return finalSizePmf / np.sum(finalSizePmf)

# Exact likelihood (Text S1) of infection times observed up to tObs with hidden
# removals, for each R0 in r0Grid, with a single initially infected premises.
#
# This gives the same value as the matrix form in Text S1, but costs O(K^2) for K
# observed infections and does not overflow for outbreaks of thousands of
# infections. Expanding the number infectious at each observed infection over the
# possible infectors gives a sum, over assignments of an infector to every
# infection, of products of one factor per infective (infectious periods are
# independent Exp(gamma)). An infective whose last assigned infection is at time L
# survives every interval up to L without causing an unobserved infection, then
# causes no further infection before tObs (noFurther). Summing over assignments by
# the number, n, of "open" infectives (with at least one infection still to cause)
# gives a forward recursion with only the moves n -> n - 1, n, n + 1.
def infectionLogLikelihoodGrid(infectionTimes, tObs, r0Grid, gamma, nPop):
    nObserved = len(infectionTimes)
    nGrid = len(r0Grid)
    pairInfectionRate = r0Grid * gamma / (nPop - 1)

    # Interval k runs from infection k to infection k + 1 (from time 0, to tObs).
    intervals = np.diff(np.concatenate(([0.0], infectionTimes, [tObs])))
    susceptible = nPop - 1 - np.arange(nObserved + 1)
    perInfectiveRate = gamma + pairInfectionRate[None, :] * susceptible[:, None]
    exponent = perInfectiveRate * intervals[:, None]
    logStillInfectious = -exponent
    stillInfectious = np.exp(logStillInfectious)
    recoveredProbability = gamma * (-np.expm1(-exponent)) / perInfectiveRate
    noFurther = np.ones((nObserved + 2, nGrid))
    for k in range(nObserved, -1, -1):
        noFurther[k] = recoveredProbability[k] + stillInfectious[k] * noFurther[k + 1]

    # The index premises either causes no infection (closes) or is open.
    weights = np.stack([noFurther[0], np.ones(nGrid)], axis=1)
    weights = weights[:, : min(1, nObserved) + 1]
    scale = weights.max(axis=1)
    weights = weights / scale[:, None]
    logLikelihood = np.log(scale)

    with np.errstate(divide="ignore"):
        for k in range(1, nObserved + 1):
            width = weights.shape[1]
            nOpen = np.arange(width, dtype=float)

            # Infection k needs an open infective. Each open infective survives
            # interval k - 1 (in logs, to avoid underflow over long gaps).
            weights[:, 0] = 0.0
            logWeights = np.log(weights) + nOpen[None, :] * logStillInfectious[k - 1][:, None]
            shift = logWeights.max(axis=1)
            finite = np.isfinite(shift)
            weights = np.zeros_like(weights)
            weights[finite] = np.exp(logWeights[finite] - shift[finite, None])
            logLikelihood[finite] += shift[finite]
            logLikelihood[~finite] = -np.inf

            # One of the n open infectives causes infection k; it and the newly
            # infected premises each close (noFurther) or stay open. More open
            # infectives than infections remaining cannot all close, so are dropped.
            newWidth = min(width + 1, nObserved - k + 1)
            chosen = weights * nOpen[None, :]
            closes = noFurther[k][:, None]
            updated = np.zeros((nGrid, newWidth))
            updated[:, : min(newWidth, width - 1)] += (chosen[:, 1:] * closes * closes)[
                :, :newWidth
            ]
            updated[:, : min(newWidth, width)] += (2.0 * chosen * closes)[:, :newWidth]
            if newWidth > 1:
                updated[:, 1:newWidth] += chosen[:, : newWidth - 1]
            updated *= (pairInfectionRate * susceptible[k - 1])[:, None]

            scale = updated.max(axis=1)
            possible = scale > 0.0
            updated[possible] /= scale[possible, None]
            updated[~possible] = 0.0
            logLikelihood[possible] += np.log(scale[possible])
            logLikelihood[~possible] = -np.inf
            weights = updated

    allClosed = weights[:, 0]
    logLikelihoodOut = np.full(nGrid, -np.inf)
    valid = (allClosed > 0.0) & np.isfinite(logLikelihood)
    logLikelihoodOut[valid] = logLikelihood[valid] + np.log(allClosed[valid])
    return logLikelihoodOut

# Exact lattice form of W1 for final-size fractions {0, 1/N, ..., 1}, along the last axis.
def histogramWasserstein1(histogramA, histogramB):
    nPop = histogramA.shape[-1] - 1
    probabilityA = histogramA / np.sum(histogramA, axis=-1, keepdims=True)
    probabilityB = histogramB / np.sum(histogramB, axis=-1, keepdims=True)
    cdfA = np.cumsum(probabilityA, axis=-1)
    cdfB = np.cumsum(probabilityB, axis=-1)
    cdfGap = np.abs(cdfA[..., :nPop] - cdfB[..., :nPop])
    return np.sum(cdfGap, axis=-1) / nPop


def main():
    with np.load(LANDSCAPE) as data:
        points, countyIndices = data["points_km"], data["county_index"]

    # beta is set for the whole of Great Britain, but the county is simulated on its
    # own premises (premises k of the county is members[k]), so transmission across
    # the county border is neglected.
    gbKernelSums = kernelSums(points, SIGMA, CUTOFF)
    beta = calibrateBeta(gbKernelSums, R0, GAMMA)
    members = np.flatnonzero(countyIndices == COUNTY)
    nPop = len(members)
    kernel = buildKernel(points[members], SIGMA, CUTOFF)
    countyR0 = np.mean(beta / GAMMA * kernel["kernelSums"])  # Eq. 9
    borderKernelFraction = 1.0 - np.sum(kernel["kernelSums"]) / np.sum(
        gbKernelSums[members]
    )

    # Index premises drawn uniformly without replacement (every premises in counties
    # with fewer than N_INDEX), each with an equal share of the outbreaks.
    indexPremises = np.sort(
        np.random.default_rng([SEED, 0, COUNTY]).choice(
            nPop, size=min(N_INDEX, nPop), replace=False
        )
    )
    nRepeats = int(np.ceil(N_REFERENCE / len(indexPremises)))
    nPerIndex = int(np.ceil(N_OBSERVATION_POOL / len(indexPremises)))
    referenceRng, observationRng, dataRng = [
        np.random.default_rng(int(seed))
        for seed in np.random.default_rng([SEED, 1, COUNTY]).integers(
            0, 2**63 - 1, size=3, dtype=np.int64
        )
    ]

    # Without calibration: the well-mixed model at the county R0.
    spatialHistogram = simulateIndexHistograms(
        kernel, beta, GAMMA, indexPremises, nRepeats, referenceRng
    ).sum(axis=0)
    wellMixedPmf = exactWellMixedFinalSizePmf(nPop, countyR0)
    w1Matched = histogramWasserstein1(spatialHistogram, wellMixedPmf)

    # Calibration to the infection times of n outbreaks from the pool, and in the
    # limit of many outbreaks (the pseudo-true R0).
    outbreaks = simulateObservationPool(
        kernel, beta, GAMMA, indexPremises, nPerIndex, T_OBS, observationRng
    )
    logLikelihoods = np.array(
        [infectionLogLikelihoodGrid(t, T_OBS, R0_GRID, GAMMA, nPop) for t in outbreaks]
    )
    wellMixedPmfs = np.array([exactWellMixedFinalSizePmf(nPop, r0) for r0 in R0_GRID])
    pseudoTrue = logLikelihoods.mean(axis=0).argmax()
    w1PseudoTrue = histogramWasserstein1(spatialHistogram, wellMixedPmfs[pseudoTrue])

    # Independent datasets of n outbreaks, with a uniform prior over R0_GRID.
    w1 = np.empty(N_DATASETS)
    modeR0 = np.empty_like(w1)
    for dataset in range(N_DATASETS):
        indices = dataRng.choice(len(outbreaks), size=N_OBS, replace=False)
        jointLogLikelihood = logLikelihoods[indices].sum(axis=0)
        posterior = np.exp(jointLogLikelihood - logsumexp(jointLogLikelihood))
        w1[dataset] = histogramWasserstein1(spatialHistogram, posterior @ wellMixedPmfs)
        modeR0[dataset] = R0_GRID[posterior.argmax()]
    np.savez(
        f"county_{COUNTY}.npz",
        county=COUNTY,
        nPop=nPop,
        beta=beta,
        countyR0=countyR0,
        borderKernelFraction=borderKernelFraction,
        spatialHistogram=spatialHistogram,
        wellMixedHistogram=wellMixedPmf,
        w1Matched=w1Matched,
        nObs=N_OBS,
        w1=w1,
        posteriorModeR0=modeR0,
        meanDeltaCalibration=(w1 - w1Matched).mean(),
        pseudoTrueR0=R0_GRID[pseudoTrue],
        w1PseudoTrue=w1PseudoTrue,
    )


if __name__ == "__main__":
    main()
