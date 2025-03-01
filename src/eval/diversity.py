"""Module to evaluate the diversity of a text dataset, using Self-BLEU, Expectation-Adjusted-Distinct,."""

import evaluate


def fleiss_kappa(ratings):
    """Calculate Fleiss' Kappa for a set of ratings (nominal categories). This.

    Args:
        ratings (list): A list of ratings for each category.

    Returns:
        float: The Fleiss' Kappa score.
    """
    n = len(ratings)
    k = len(ratings[0])
    N = n * k
    p = [0] * k
    P = [0] * n
    for i in range(n):
        for j in range(k):
            p[j] += ratings[i][j]
            P[i] += ratings[i][j] ** 2
    P = [(p_i - 1) / (n - 1) for p_i in P]
    P_bar = sum(P) / n
    P_e = sum([p_j**2 for p_j in p]) / N
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa
