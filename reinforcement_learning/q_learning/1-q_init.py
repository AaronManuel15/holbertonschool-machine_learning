"""Task 1. Initialize the Q-table"""
import numpy as np


def q_init(env):
    """Initialize the Q-table"""
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
