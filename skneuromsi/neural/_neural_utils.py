import itertools
import numpy as np


def calculate_two_peaks_probability(peaks_values):
    combinations = list(
        itertools.chain.from_iterable(
            itertools.combinations(peaks_values, i + 2)
            for i in range(len(peaks_values))
        )
    )

    probs_array = np.array([])

    for i in combinations:
        probs_array = np.append(probs_array, np.array(i).prod())

    return probs_array.sum() / probs_array.size
