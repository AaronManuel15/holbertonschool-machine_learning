"""Task 2. Epsilon Greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses Epsilon Greedy to Determine the next action.
    Args:
        Q: numpy.ndarray containing the q-table
        state: Current state
        epsilon: epsilon to use for the calculation
    Returns:
        the next action index"""

    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(3)
    else:
        return np.argmax(Q[state])
