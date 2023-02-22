#!/usr/bin/env python3
"""Task 11"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay:
    alpha: is the original learning rate
    decay_rate: is the weight used to determine the rate at which
        alpha will decay
    global_step: is the number of passes of gradient descent that
        have elapsed
    decay_step: is the number of passes of gradient descent that
        should occur before alpha is decayed further

    Returns: the updated value for alpha"""

    return alpha/(1 + decay_rate*(global_step//decay_step))
