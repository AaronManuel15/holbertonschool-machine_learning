#!/usr/bin/env python3
"""Task 7"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if gradient descent should be stopped early:

    cost: current validation cost of the neural network
    opt_cost: lowest recorded validation cost of the neural network
    threshold: threshold used for early stopping
    patience: patience count used for early stopping
    count: count of how long the threshold has not been met

    Returns: a boolean of whether the network should be stopped early,
        followed by the updated count"""

    if cost < opt_cost - threshold:
        return (False, 0)
    elif count + 1 >= patience:
        return (True, count + 1)
    else:
        return (False, count + 1)
