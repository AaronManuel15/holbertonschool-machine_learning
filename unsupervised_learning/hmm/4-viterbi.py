#!/usr/bin/env python3
"""Task 4: The Viretbi Algorithm"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states for a hidden
    markov model
    Args:
        Observation: is a numpy.ndarray of shape (T,) that contains the index
            of the observation
            T is the number of observations
        Emission: is a numpy.ndarray of shape (N, M) containing the emission
            probability of a specific observation given a hidden state
            Emission[i, j] is the probability of observing j given the hidden
            state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition: is a 2D numpy.ndarray of shape (N, N) containing the
            transition probabilities
            Transition[i, j] is the probability of transitioning from the
            hidden state i to j
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
            starting in a particular hidden state
        Returns:
            path, P, or None, None on failure
            path is the a list of length T containing the most likely sequence
                of hidden states
            P is the probability of obtaining the path sequence"""

    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    V = np.zeros((N, T))
    B = np.zeros((N, T))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for s in range(N):
            V[s, t] = np.max(V[:, t - 1] * Transition[:, s] *
                             Emission[s, Observation[t]])
            B[s, t] = np.argmax(V[:, t - 1] * Transition[:, s])

    path = [np.argmax(V[:, T - 1])]
    for t in range(T - 1, 0, -1):
        path.append(int(B[path[-1], t]))
    path.reverse()

    P = np.max(V[:, T - 1])

    return path, P
