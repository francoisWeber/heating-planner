import numpy as np


def comparison_neutral_delta(value_minus_reference, normalize=True):
    delta_compared = -np.abs(value_minus_reference)
    if normalize:
        delta_compared /= np.nanstd(delta_compared)
    return delta_compared


def comparison_upper_better_delta(value_minus_reference, decay=0.1, normalize=True):
    delta_compared = np.where(
        value_minus_reference > 0, value_minus_reference * decay, value_minus_reference
    )
    if normalize:
        delta_compared /= np.nanstd(delta_compared)
    return delta_compared


def comparison_lower_better_delta(value_minus_reference, decay=0.1, normalize=True):
    return comparison_upper_better_delta(
        -value_minus_reference, decay=decay, normalize=normalize
    )
