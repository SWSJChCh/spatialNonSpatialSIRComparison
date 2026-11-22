import numpy as np

# Hosts are sampled uniformly on a periodic square landscape.
def buildGraphs(nGraphs, nPop, boxSize=1.0, rng=None):
    rng = np.random.default_rng(rng)
    points = rng.uniform(low=0.0, high=boxSize, size=(nGraphs, nPop, 2))
    initialHosts = rng.integers(low=0, high=nPop, size=nGraphs)
    return {
        "points": points,
        "initialHosts": initialHosts,
        "boxSize": boxSize,
    }

# Exponential kernel with periodic (minimum-image) distances.
def spatialKernel(points, sigma, boxSize):
    halfBox = 0.5 * boxSize
    x, y = points[:, 0], points[:, 1]
    dx = (x[:, np.newaxis] - x[np.newaxis, :] + halfBox) % boxSize - halfBox
    dy = (y[:, np.newaxis] - y[np.newaxis, :] + halfBox) % boxSize - halfBox
    distance = np.sqrt(dx**2 + dy**2)
    kernel = np.exp(-distance / sigma)
    np.fill_diagonal(kernel, 0.0)
    return kernel

# Match the mean initial infection pressure, in units of gamma, to r0.
def calibrateBeta(graphs, sigma, r0, gamma):
    boxSize = graphs["boxSize"]
    kernelSums = []
    for points, initialHost in zip(graphs["points"], graphs["initialHosts"]):
        kernel = spatialKernel(points, sigma, boxSize)
        kernelSums.append(np.sum(kernel[:, initialHost]))
    meanKernelSum = np.mean(kernelSums)
    beta = r0 * gamma / meanKernelSum
    return beta

# For the final size distribution, we use the embedded event chain (as infection times are not required).
# Please note that this gives the same average results as explicit simulations of the full Gillespie
# algorithm.
def spatialFinalSize(infectionRates, gamma, initialHost, rng):
    nPop = len(infectionRates)
    susceptible = np.ones(nPop, dtype=bool)
    infected = np.zeros(nPop, dtype=bool)
    susceptible[initialHost] = False
    infected[initialHost] = True
    infectedCount = 1
    pressure = infectionRates[:, initialHost] * susceptible

    while infectedCount > 0:
        totalInfectionRate = np.sum(pressure)
        totalRecoveryRate = gamma * infectedCount
        totalEventRate = totalInfectionRate + totalRecoveryRate
        nextIsInfection = (
            totalInfectionRate > 0.0
            and rng.random() < totalInfectionRate / totalEventRate
        )

        if nextIsInfection:
            cumulativePressure = np.cumsum(pressure)
            target = rng.random() * cumulativePressure[-1]
            newCase = np.searchsorted(cumulativePressure, target, side="right")
            susceptible[newCase] = False
            infected[newCase] = True
            infectedCount += 1
            pressure += infectionRates[:, newCase]
        else:
            infectedHosts = np.flatnonzero(infected)
            recovered = infectedHosts[rng.integers(infectedCount)]
            infected[recovered] = False
            infectedCount -= 1
            pressure -= infectionRates[:, recovered]
            np.maximum(pressure, 0.0, out=pressure)

        pressure *= susceptible

    return nPop - np.sum(susceptible)

def simulateGraphHistograms(graphs, sigma, beta, gamma, nRepeats, rng=None):
    rng = np.random.default_rng(rng)
    nGraphs, nPop = graphs["points"].shape[:2]
    boxSize = graphs["boxSize"]
    histograms = np.zeros((nGraphs, nPop + 1), dtype=int)
    for graphIndex in range(nGraphs):
        points = graphs["points"][graphIndex]
        initialHost = graphs["initialHosts"][graphIndex]
        infectionRates = beta * spatialKernel(points, sigma, boxSize)
        for _ in range(nRepeats):
            finalSize = spatialFinalSize(infectionRates, gamma, initialHost, rng)
            histograms[graphIndex, finalSize] += 1
    return histograms

# Propagate probability through the exact finite-population well-mixed jump chain.
def exactWellMixedFinalSizePmf(nPop, r0):
    stateProbability = np.zeros((nPop + 1, nPop + 1))
    stateProbability[nPop - 1, 1] = 1.0
    for susceptible in range(nPop - 1, -1, -1):
        relativeInfectionRate = r0 * susceptible
        relativeRecoveryRate = nPop - 1
        infectionProbability = relativeInfectionRate / (
            relativeInfectionRate + relativeRecoveryRate
        )
        recoveryProbability = 1.0 - infectionProbability
        maxInfected = nPop - susceptible
        for infected in range(maxInfected, 0, -1):
            mass = stateProbability[susceptible, infected]
            if susceptible > 0:
                stateProbability[susceptible - 1, infected + 1] += (
                    mass * infectionProbability
                )
            stateProbability[susceptible, infected - 1] += mass * recoveryProbability
    finalSizes = np.arange(nPop + 1)
    return stateProbability[nPop - finalSizes, 0]

def exceedanceFromHistogram(histogram, xGrid):
    nPop = histogram.shape[-1] - 1
    finalSizeFractions = np.arange(nPop + 1) / nPop
    totalCount = np.sum(histogram, axis=-1, keepdims=True)
    cdf = np.cumsum(histogram, axis=-1) / totalCount
    largestSizeAtOrBelowX = (
        np.searchsorted(finalSizeFractions, xGrid, side="right") - 1
    )
    exceedance = 1.0 - np.take(cdf, largestSizeAtOrBelowX, axis=-1)
    return np.where(largestSizeAtOrBelowX >= 0, exceedance, 1.0)

# Exact lattice form of W1 for final-size fractions {0, 1/N, ..., 1}, along the last axis.
def histogramWasserstein1(histogramA, histogramB):
    nPop = histogramA.shape[-1] - 1
    probabilityA = histogramA / np.sum(histogramA, axis=-1, keepdims=True)
    probabilityB = histogramB / np.sum(histogramB, axis=-1, keepdims=True)
    cdfA = np.cumsum(probabilityA, axis=-1)
    cdfB = np.cumsum(probabilityB, axis=-1)
    cdfGap = np.abs(cdfA[..., :nPop] - cdfB[..., :nPop])
    return np.sum(cdfGap, axis=-1) / nPop

def runCell(graphs, sigma, beta, gamma, nRepeats, r0, xGrid, rng=None):
    nPop = graphs["points"].shape[1]
    perGraphHistograms = simulateGraphHistograms(
        graphs, sigma, beta, gamma, nRepeats, rng
    )
    spatialHistogram = np.sum(perGraphHistograms, axis=0)
    wellMixedHistogram = exactWellMixedFinalSizePmf(nPop, r0)
    return {
        "spatialHistogram": spatialHistogram,
        "wellMixedHistogram": wellMixedHistogram,
        "meanExceedance": exceedanceFromHistogram(spatialHistogram, xGrid),
        "wellMixedExceedance": exceedanceFromHistogram(wellMixedHistogram, xGrid),
        "w1Pooled": histogramWasserstein1(spatialHistogram, wellMixedHistogram),
    }

# Infection times for the likelihood with hidden removals.
def spatialInfectionTimes(infectionRates, gamma, initialHost, rng):
    nPop = len(infectionRates)
    susceptible = np.ones(nPop, dtype=bool)
    infected = np.zeros(nPop, dtype=bool)
    susceptible[initialHost] = False
    infected[initialHost] = True
    infectedCount = 1
    pressure = infectionRates[:, initialHost] * susceptible
    time = 0.0
    infectionTimes = []

    while infectedCount > 0:
        totalInfectionRate = np.sum(pressure)
        totalRecoveryRate = gamma * infectedCount
        totalEventRate = totalInfectionRate + totalRecoveryRate
        waitingTime = rng.exponential(scale=1.0 / totalEventRate)
        time += waitingTime
        nextIsInfection = (
            totalInfectionRate > 0.0
            and rng.random() < totalInfectionRate / totalEventRate
        )

        if nextIsInfection:
            cumulativePressure = np.cumsum(pressure)
            target = rng.random() * cumulativePressure[-1]
            newCase = np.searchsorted(cumulativePressure, target, side="right")
            susceptible[newCase] = False
            infected[newCase] = True
            infectedCount += 1
            pressure += infectionRates[:, newCase]
            infectionTimes.append(time)
        else:
            infectedHosts = np.flatnonzero(infected)
            recovered = infectedHosts[rng.integers(infectedCount)]
            infected[recovered] = False
            infectedCount -= 1
            pressure -= infectionRates[:, recovered]
            np.maximum(pressure, 0.0, out=pressure)

        pressure *= susceptible

    return np.array(infectionTimes)

# One outbreak, observed up to tObs, on each of poolSize distinct landscapes.
def simulateObservationPool(graphs, sigma, beta, gamma, tObs, poolSize, rng=None):
    rng = np.random.default_rng(rng)
    nGraphs = len(graphs["points"])
    boxSize = graphs["boxSize"]
    chosenGraphs = rng.choice(nGraphs, size=poolSize, replace=False)
    observations = []
    for graphIndex in chosenGraphs:
        points = graphs["points"][graphIndex]
        initialHost = graphs["initialHosts"][graphIndex]
        infectionRates = beta * spatialKernel(points, sigma, boxSize)
        infectionTimes = spatialInfectionTimes(infectionRates, gamma, initialHost, rng)
        observedTimes = infectionTimes[infectionTimes <= tObs]
        observations.append(observedTimes)
    return observations

# Exact likelihood (Text S1) of infection times observed up to tObs with hidden
# removals, for each R0 in r0Grid. 
# 
# Please note that for around 1,000 infectionsor more this function will overflow numerically.
def infectionLogLikelihoodGrid(infectionTimes, tObs, r0Grid, gamma, nPop):
    nObserved = len(infectionTimes)
    nStates = nObserved + 2
    binomial = np.zeros((nStates, nStates))
    binomial[:, 0] = 1.0
    for n in range(1, nStates):
        binomial[n, 1 : n + 1] = binomial[n - 1, :n] + binomial[n - 1, 1 : n + 1]
    powers = np.arange(nStates)
    pairInfectionRate = r0Grid * gamma / (nPop - 1)
    mass = np.zeros((len(r0Grid), nStates))
    mass[:, 1] = 1.0
    logLikelihood = np.zeros(len(r0Grid))
    previousTime = 0.0
    eventTimes = [*infectionTimes, tObs]

    for infections, eventTime in enumerate(eventTimes):
        susceptible = nPop - 1 - infections
        interval = eventTime - previousTime
        perInfectiveRate = gamma + pairInfectionRate * susceptible
        stillInfectiousProbability = np.exp(-perInfectiveRate * interval)
        recoveredProbability = (
            gamma * (-np.expm1(-perInfectiveRate * interval)) / perInfectiveRate
        )
        stillInfectiousPowers = stillInfectiousProbability[:, None] ** powers
        recoveredPowers = recoveredProbability[:, None] ** powers
        propagated = np.empty_like(mass)
        for remaining in range(nStates):
            waysToChoose = binomial[remaining:, remaining]
            othersRecovered = recoveredPowers[:, : nStates - remaining]
            remainingStayed = stillInfectiousPowers[:, remaining]
            propagated[:, remaining] = remainingStayed * np.sum(
                mass[:, remaining:] * waysToChoose * othersRecovered, axis=1
            )
        noInfectionProbability = np.sum(propagated, axis=1)
        logLikelihood += np.log(noInfectionProbability)
        mass = propagated / noInfectionProbability[:, None]
        if infections == nObserved:
            return logLikelihood

        # One of the i infectives causes the observed infection.
        infectivesBefore = np.arange(nStates - 1.0)
        jump = np.zeros_like(mass)
        jump[:, 1:] = (
            pairInfectionRate[:, None] * susceptible * infectivesBefore * mass[:, :-1]
        )
        infectionDensity = np.sum(jump, axis=1)
        logLikelihood += np.log(infectionDensity)
        mass = jump / infectionDensity[:, None]
        previousTime = eventTime

# W1 of the pooled histogram, with a bootstrap standard error that resamples
# whole landscapes (100 replicates at a time to bound memory).
def clusteredWasserstein1WithSe(
    graphHistograms, referenceHistogram, nBootstrap, rng=None
):
    rng = np.random.default_rng(rng)
    graphHistograms = graphHistograms.astype(float)
    nGraphs = len(graphHistograms)
    batchSize = 100
    equalProbabilities = np.full(nGraphs, 1.0 / nGraphs)
    bootstrapW1 = []
    for start in range(0, nBootstrap, batchSize):
        nReplicates = min(batchSize, nBootstrap - start)
        timesDrawn = rng.multinomial(nGraphs, equalProbabilities, size=nReplicates)
        resampledHistograms = timesDrawn @ graphHistograms
        bootstrapW1.extend(
            histogramWasserstein1(resampledHistograms, referenceHistogram)
        )
    pooledHistogram = np.sum(graphHistograms, axis=0)
    w1 = histogramWasserstein1(pooledHistogram, referenceHistogram)
    standardError = np.std(bootstrapW1, ddof=1)
    return w1, standardError

# Posterior over the R0 grid (flat prior) and W1 between the spatial and
# posterior-predictive final-size distributions, for the nested datasets made of
# the first n outbreaks in `indices`, for each n in nObs.
def evaluateNestedSets(
    indices, nObs, logLikelihoodPool, wellMixedPmfGrid, spatialHistogram
):
    w1, logLikelihood, posterior = [], [], []
    jointLogLikelihood = 0.0
    previousCounts = [0, *nObs[:-1]]
    for previousCount, count in zip(previousCounts, nObs):
        newOutbreaks = indices[previousCount:count]
        jointLogLikelihood = jointLogLikelihood + np.sum(
            logLikelihoodPool[newOutbreaks], axis=0
        )
        relativePosterior = np.exp(jointLogLikelihood - np.max(jointLogLikelihood))
        posteriorWeights = relativePosterior / np.sum(relativePosterior)
        predictivePmf = posteriorWeights @ wellMixedPmfGrid
        w1.append(histogramWasserstein1(spatialHistogram, predictivePmf))
        logLikelihood.append(jointLogLikelihood)
        posterior.append(posteriorWeights)
    return np.array(w1), np.array(logLikelihood), np.array(posterior)

# Pointwise central 95% interval of curves (one per row) weighted by `weights`.
def weightedCurveInterval(curves, weights):
    order = np.argsort(curves, axis=0)
    sortedCurves = np.take_along_axis(curves, order, axis=0)
    cumulativeWeight = np.cumsum(weights[order], axis=0)
    columns = np.arange(curves.shape[1])
    lowerQuantileRow = np.sum(cumulativeWeight < 0.025, axis=0)
    upperQuantileRow = np.sum(cumulativeWeight < 0.975, axis=0)
    lowerCurve = sortedCurves[lowerQuantileRow, columns]
    upperCurve = sortedCurves[upperQuantileRow, columns]
    return np.array([lowerCurve, upperCurve])

# Deterministic limit. With probability 1 - 1/r0 a major outbreak infects the
# fraction z solving z = 1 - exp(-r0 z). (We do not use this but it can be informative). 
def odeExceedance(r0, xGrid):
    if r0 <= 1.0:
        return np.zeros_like(xGrid)
    majorOutbreakProbability = 1.0 - 1.0 / r0
    lower, upper = 0.0, 1.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        rightHandSide = -np.expm1(-r0 * middle)
        if rightHandSide > middle:
            lower = middle
        else:
            upper = middle
    majorOutbreakFraction = 0.5 * (lower + upper)
    return np.where(xGrid < majorOutbreakFraction, majorOutbreakProbability, 0.0)