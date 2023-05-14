import functools
import logging
import math
import networkx as nx
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as spg
import sys


def wassersteinMinCostFlow(mu, nu, dist, measDenom, distDenom):
    """
    This computes the (rounded) Wasserstein distance of two measures via a min-cost-flow algorithm of SciPy.
    The metric space is {0, ..., n - 1}, where n is the length of mu (and nu).
    The measures and distance matrix are represented by fractions,
    where mu, nu, and dist contain the nominators and measDenom and distDenom are the denominators.
    The return value is also represented as a fraction (w.r.t. distDenom) and rounded to an integer.

    :param mu: Array of natural numbers representing the first measure. Element i has mass mu[i] / measDenom.
    :param nu: Array of natural numbers representing the second measure. Same length as mu.
    :param dist: Matrix of natural numbers representing distances between elements of the metric space.
                 Elements i and j have distance dist[i, j] / distDenom.
    :param measDenom: A natural number that acts as the denominator for the measures,
                      i.e., mu / measDenom and nu / measDenom are the actual measures of total mass at most one.
    :param distDenom: A natural number that acts as the denominator for the distances,
                      i.e., dist / distDenom is the actual distance matrix with distances at most 1.
    :returns: Natural number such that the return value / distDenom is the (approximate) distance between the two measures.
    """
    assert(mu.size == nu.size)
    if sum(mu) < sum(nu):
        mu, nu = nu, mu

    assert(sum(mu) >= sum(nu))
    n = mu.size

    G = nx.DiGraph()
    for i in range(0, n):
        if mu[i] > 0:
            G.add_node(i, demand = - mu[i])

        if nu[i] > 0:
            G.add_node(n + i, demand = nu[i])

    G.add_node(2 * n, demand = sum(mu) - sum(nu))

    for i in range(0,n):
        if mu[i] > 0:
            G.add_edge(i, 2 * n, weight = distDenom)

            for j in range(0, n):
                if nu[j] > 0:
                    G.add_edge(i, n + j, weight = dist[i, j])

    flow_dict = nx.min_cost_flow(G)
    cost = nx.cost_of_flow(G, flow_dict) / measDenom

    # the resulting distance might not be an integer
    return np.rint(cost).astype(np.int64)


def wassersteinLP(mu, nu, dist, measDenom, distDenom):
    """
    The same as wassersteinMinCostFlow but computes the distance by solving an LP (via SciPy) instead.
    Slower than wassersteinMinCostFlow.
    """
    assert(mu.size == nu.size)
    if sum(mu) < sum(nu):
        mu, nu = nu, mu

    assert(sum(mu) >= sum(nu))
    n = mu.size

    c = dist.flatten()
    bub = mu
    Aub = np.zeros(shape = (n, n * n), dtype = np.int64)
    for i in range(n):
        Aub[i, i * n : (i + 1) * n] = np.ones(n, dtype = np.int64)

    beq = nu
    Aeq = np.zeros(shape = (n, n * n), dtype = np.int64)
    for j in range(n):
        for h  in range(n):
            Aeq[j, j + h * n] = 1

    res = linprog(c, A_ub = Aub, b_ub = bub, A_eq = Aeq, b_eq = beq, bounds = (0, None))
    cost = ((sum(mu) - sum(nu)) * distDenom + res.fun) / measDenom

    return np.rint(cost).astype(np.int64)


def betaFlowSciPy(mu, nu, dist, alpha, measDenom, distDenom):
    """
    Function for computing the beta function used in the prokhorov metric.
    The result is computed via a max-flow algorithm of SciPy.
    See the documentation of prokhorov for an explanation.

    :param mu: Array of natural numbers representing the first measure. Element i has mass mu[i] / measDenom.
    :param nu: Array of natural numbers representing the second measure. Same length as mu.
    :param dist: Matrix of natural numbers representing distances between elements of the metric space.
                 Elements i and j have distance dist[i, j] / distDenom.
    :param alpha: Natural number representing the alpha parameter. The actual value is alpha / distDenom.
    :param measDenom: A natural number that acts as the denominator for the measures,
                      i.e., mu / measDenom and nu / measDenom are the actual measures of total mass at most one.
    :param distDenom: A natural number that acts as the denominator for the distances,
                      i.e., dist / distDenom is the actual distance matrix with distances at most 1.
    :returns: Natural number such that the return value / distDenom is the value beta(alpha / distDenom).
    """
    assert(sum(mu) >= sum(nu))
    assert(mu.size == nu.size)
    n = mu.size

    A = np.zeros(shape = (2 * n + 2, 2 * n + 2), dtype = np.int64)
    # 2n is s
    # 2n + 1 is t

    for i in range(0, n):
        A[2 * n, i] = mu[i]
        A[n + i, 2 * n + 1] = nu[i]

    for i in range(n):
        for j in range(n):
            if dist[i, j] <= alpha:
                A[i, n + j] = measDenom

    graph = csr_matrix(A)
    res = spg.maximum_flow(graph, 2 * n, 2 * n + 1)

    return (sum(mu) - res.flow_value) * (distDenom // measDenom)


def betaFlowNX(mu, nu, dist, alpha, measDenom, distDenom):
    """
    The same as betaFlowSciPy but computes the function by a max-flow algorithm of NetworkX instead.
    Slower than betaFlowSciPy.
    """
    assert(sum(mu) >= sum(nu))
    assert(mu.size == nu.size)
    n = mu.size

    G = nx.DiGraph()
    G.add_node("s")
    G.add_node("t")
    for i in range(0, n):
        if mu[i] > 0:
            G.add_node(i)
            G.add_edge("s", i, capacity = mu[i])

        if nu[i] > 0:
            G.add_node(n + i)
            G.add_edge(n + i, "t", capacity = nu[i])

    for i in range(0,n):
        for j in range(n):
            if dist[i, j] <= alpha:
                G.add_edge(i, n + j, capacity = measDenom)

    flow_value, flow_dict = nx.maximum_flow(G, "s", "t")

    return (sum(mu) - flow_value) * (distDenom // measDenom)


def betaLP(mu, nu, dist, alpha, measDenom, distDenom):
    """
    The same as betaFlowSciPy but computes the function by solving an LP (via SciPy) instead.
    Slower than betaFlowSciPy.
    """
    assert(sum(mu) >= sum(nu))
    assert(mu.size == nu.size)
    n = mu.size

    f = lambda x : -1 if x <= alpha else 0
    vf = np.vectorize(f)
    c = vf(dist.flatten())

    Aub = np.zeros(shape = (n, n * n), dtype = np.int64)
    for i in range(n):
        Aub[i, i * n : (i + 1) * n] = np.ones(n, dtype = np.int64)
    bub = mu

    Aeq = np.zeros(shape = (n, n * n), dtype = np.int64)
    for i in range(n):
        for j  in range(n):
            Aeq[i, i + j * n] = 1
    beq = nu

    res = linprog(c, A_ub = Aub, b_ub = bub, A_eq = Aeq, b_eq = beq, bounds = (0, None))

    return (sum(mu) + np.rint(res.fun).astype(np.int64)) * (distDenom // measDenom)


def prokhorov(mu, nu, dist, measDenom, distDenom, beta = betaFlowSciPy):
    """
    This computes the Prokhorov distance of two measures. via a min-cost-flow algorithm.
    The metric space is {0, ..., n - 1}, where n is the length of mu (and nu).
    The measures and distance matrix are represented by fractions,
    where mu, nu, and dist contain the nominators and measDenom and distDenom are the denominators.
    distDenom is assumed to be a multiple of measDenom.

    The distance is computed using the fact that prokhorov(mu, nu) = inf{alpha > 0 | alpha >= beta(alpha)},
    where a the function for computing beta is given as an argument to this function.

    The return value is also represented as a fraction (w.r.t. distDenom);
    note that this is exact as not rounding is necessary.

    :param mu: Array of natural numbers representing the first measure. Element i has mass mu[i] / measDenom.
    :param nu: Array of natural numbers representing the second measure. Same length as mu.
    :param dist: Matrix of natural numbers representing distances between elements of the metric space.
                 Elements i and j have distance dist[i, j] / distDenom.
    :param measDenom: A natural number that acts as the denominator for the measures,
                      i.e., mu / measDenom and nu / measDenom are the actual measures of total mass at most one.
    :param distDenom: A natural number that acts as the denominator for the distances,
                      i.e., dist / distDenom is the actual distance matrix with distances at most 1.
    :param beta: Function for computing the beta function.
    :returns: Natural number such that the return value / distDenom is the distance between the two measures.
    """
    assert(mu.size == nu.size)
    if sum(mu) < sum(nu):
        mu, nu = nu, mu

    assert(sum(mu) >= sum(nu))

    allDistances = set(dist.flatten())
    allDistances.add(0)

    def f(alpha):
        betaAlpha = beta(mu, nu, dist, alpha, measDenom, distDenom)
        return max(alpha, betaAlpha)

    return min(map(f, allDistances))


def computeDistance(G, H, metric = wassersteinMinCostFlow, iterationBound = math.inf, convIterationBound = math.inf):
    """
    This computes the distance of two graphs by using the provided metric on measures.
    It runs color refinement on both graphs in parallel and, after every refinement round,
    computes a matrix of distances between all colors.
    In the end, the color distributions of both graphs are compared
    and the distance is returned.

    :param G: The first graph.
    :param H: The second graph.
    :param metric: The metric to use for comparing measures.
    :param iterationBound: The maximum number of iterations performed.
                           Note that, even after having obtained a stable coloring,
                           additional iterations might be needed since the distance between colors might still grow.
                           This is a bound on the total number of iterations.
    :param convIterationBound: The maximum number of iterations performed after having obtained a stable coloring.
                           Note that, even after having obtained a stable coloring of the vertices,
                           additional iterations might be needed since the distance between colors might still grow.
                           This is a bound on these additional iterations.
    :returns: The distance between G and H. This is a float from [0,1].
    """
    I = nx.disjoint_union(G, H)
    orderG = len(G)
    orderH = len(H)

    colors = [0] # list of all currently present colors
    colorOf = [0] * len(I) # maps vertex v to its color (index of "colors")
    numColors = 1
    distMatrix = np.zeros(shape = (1, 1), dtype = np.int64)
    measDenom = orderG * orderH
    distDenom = measDenom * measDenom * measDenom
    # We represent our measures as fractions with denominator measDenom
    # and our distances as fractions with denominator distDenom.
    # This way, we avoid floating-point errors.
    # Moreover, most flow algorithms require integers, i.e.,
    # we avoid these conversions from floats to ints and back again.
    # distDenom might be increased (to a larger multiple of measDenom)
    # to increase the accuracy of the Wasserstein metric.
    # For the Prokhorov metric, distDenom = measDenom would suffice
    # as all numbers we obtain can be written as a fraction with denominator measDenom.

    it = 0
    convIt = 0
    converged = False
    #while not done:
    while not converged and it < iterationBound and convIt < convIterationBound:
        logging.info("Round #{0}".format(it + 1))
        it += 1

        # Compute new colors for vertices in I
        newColors = []
        newColorOf = []
        numNewColors = 0

        newColorOccursInG = []
        newColorOccursInH = []

        for v in I:
            isFromG = v < orderG
            graphSize = orderG if isFromG else orderH
            otherGraphSize = orderH if isFromG else orderG

            color = np.zeros(numColors, dtype = np.int64)
            for u in I.neighbors(v):
                color[colorOf[u]] += otherGraphSize

            indexOf = numNewColors
            for i in range(0, numNewColors):
                if np.array_equal(color, newColors[i]):
                    indexOf = i

                    if isFromG:
                        newColorOccursInG[indexOf] = True
                    else:
                        newColorOccursInH[indexOf] = True

                    break

            if indexOf == numNewColors:
                newColors.append(color)
                numNewColors += 1
                if isFromG:
                    newColorOccursInG.append(True)
                    newColorOccursInH.append(False)
                else:
                    newColorOccursInG.append(False)
                    newColorOccursInH.append(True)

            newColorOf.append(indexOf)

        # Compute new distance matrix
        newDistMatrix = np.zeros(shape = (numNewColors, numNewColors), dtype = np.int64)

        for i in range(numNewColors):
            for j in range(i + 1, numNewColors):
                if ((newColorOccursInG[i] and not newColorOccursInH[i]) and (newColorOccursInG[j] and not newColorOccursInH[j])):
                    # Both colors only occur in G, we do not need this distance
                    d = 1
                elif ((not newColorOccursInG[i] and newColorOccursInH[i]) and (not newColorOccursInG[j] and newColorOccursInH[j])):
                    # Both colors only occur in H, we do not need this distance
                    d = 1
                else:
                    d = metric(newColors[i], newColors[j], distMatrix, measDenom, distDenom)

                newDistMatrix[i, j] = d
                newDistMatrix[j, i] = d

        # Check if we need another iteration
        converged = np.array_equal(newDistMatrix, distMatrix)
        if numNewColors == numColors:
            convIt += 1

        # Update color list and distance matrix
        colors = newColors
        colorOf = newColorOf
        numColors = numNewColors
        distMatrix = newDistMatrix
        logging.info("{0} colors after this round".format(numColors))
        logging.debug("New distance matrix: {0}".format(distMatrix))

    # We have computed the colors of our vertices and their distances
    # Compute distributions for G and H
    distG = np.zeros(numColors)
    distH = np.zeros(numColors)

    for v in I:
        if v < orderG:
            distG[colorOf[v]] += orderH
        else:
            distH[colorOf[v]] += orderG

    return metric(distG, distH, distMatrix, measDenom, distDenom) / distDenom

def demoRun(G, H, description, additionalInformation, metric, iterationBound = math.inf, convIterationBound = math.inf, verbose=True):
    startTime = time.time()
    dist = computeDistance(G, H, metric, iterationBound, convIterationBound)
    elapsedTime = time.time() - startTime
    if verbose:
        print("{0}: \t{1:.5f} [{2:.2f} seconds] ({3})".format(description, dist, elapsedTime, additionalInformation))
    return dist

def demoRuns(G, H):
    demoRun(G, H, "Wasserstein distance", "flow", wassersteinMinCostFlow)
    demoRun(G, H, "Wasserstein distance", "LP", wassersteinLP)
    demoRun(G, H, "Wasserstein distance", "flow, at most 3 iterations in total", wassersteinMinCostFlow, iterationBound = 3)
    demoRun(G, H, "Wasserstein distance", "flow, at most 3 iterations after refinement", wassersteinMinCostFlow, convIterationBound = 3)
    demoRun(G, H, "Prokhorov distance", "flow", prokhorov)
    demoRun(G, H, "Prokhorov distance", "LP", functools.partial(prokhorov, beta = betaLP))


if __name__ == "__main__":
    #logging.basicConfig(stream = sys.stderr, level = logging.INFO)
    random_state = np.random.RandomState(42)

    print("6-cycle and two 3-cycles")
    G = nx.cycle_graph(6)
    C3 = nx.cycle_graph(3)
    H = nx.disjoint_union(C3, C3)
    demoRuns(G, H)
    print("")

    n = 30
    print("{0}-path and {0}-cycle".format(n))
    G = nx.path_graph(n)
    H = nx.cycle_graph(n)
    demoRuns(G, H)
    print("")

    n = 30
    print("{0}-cycle and {0}-clique".format(n))
    G = nx.cycle_graph(n)
    H = nx.complete_graph(n)
    demoRuns(G, H)
    print("")

    n = 30
    print("{0}-path and {0}-clique".format(n))
    G = nx.path_graph(n)
    H = nx.complete_graph(n)
    demoRuns(G, H)
    print("")

    n = 10
    p1 = 0.3
    p2 = 0.7
    print("{0}-vertex random graphs (p_1 = {1}, p_2 = {2})".format(n, p1, p2))
    G = nx.erdos_renyi_graph(n, p1, seed = random_state)
    H = nx.erdos_renyi_graph(n, p2, seed = random_state)
    demoRuns(G, H)
    print("")

    n = 10
    p1 = 0.5
    p2 = 0.5
    print("{0}-vertex random graphs (p_1 = {1}, p_2 = {2})".format(n, p1, p2))
    G = nx.erdos_renyi_graph(n, p1, seed = random_state)
    H = nx.erdos_renyi_graph(n, p2, seed = random_state)
    demoRuns(G, H)
    print("")
