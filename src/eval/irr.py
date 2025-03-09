"""Compute inter-annotator agreement (or inter-rater reliability, irr) for a given dataset."""


def fleiss_kappa(ratings: list[list[int]]) -> float:
    """Compute Fleiss' Kappa for given categorical ratings by fixed number of annotators.

    For N items, each of which is rated by n annotators on a fixed set of k categories, the Fleiss' Kappa score measures the agreement between the annotators.

    Args:
        ratings (list[list[int]]): A list of lists of ratings. Each list represents the ratings (nominal categories) of a single item by multiple annotators.

    Returns:
        The Fleiss' Kappa score (0 <= kappa <= 1). A kappa of 0 indicates agreement equivalent to random chance, and a kappa of 1 indicates perfect agreement.
    """
    n = len(ratings)
    k = len(ratings[0])
    N = n * k

    # Compute P_i
    P_i: list[float] = [0] * k
    for j in range(k):
        for i in range(n):
            P_i[j] += ratings[i].count(j)

    P_i = [x / N for x in P_i]

    # Compute P_i_j
    P_i_j: list[float] = [0] * (k * (k - 1) // 2)
    for j in range(k):
        for h in range(j):
            for i in range(n):
                P_i_j[j + h] += ratings[i].count(j) * ratings[i].count(h)

    P_i_j = [x / (n * (n - 1)) for x in P_i_j]

    # Compute P_e
    P_e = sum([x**2 for x in P_i])

    kappa = (sum(P_i_j) - P_e) / (1 - P_e)
    assert 0 <= kappa
    assert kappa <= 1

    return kappa


def cohen_kappa(ratings1: list[float], ratings2: list[float]) -> float:
    """Compute Cohen's Kappa for given categorical ratings on the same items, by two annotators.

    Args:
        ratings1 (list[int]): A list of ratings by the first annotator.
        ratings2 (list[int]): A list of ratings by the second annotator.

    Returns:
        The Cohen's Kappa score (-1 <= kappa <= 1). A kappa of 0 indicates agreement equivalent to random chance, and a kappa of 1 indicates perfect agreement.
    """
    assert len(ratings1) == len(ratings2)

    n = len(ratings1)

    # Compute P_o
    P_o = sum([ratings1[i] == ratings2[i] for i in range(n)]) / n

    # Compute P_e
    P_e = sum([ratings1.count(i) * ratings2.count(i) for i in set(ratings1)]) / n**2

    kappa = (P_o - P_e) / (1 - P_e)
    assert -1 <= kappa
    assert kappa <= 1

    return kappa
