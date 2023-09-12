"""Task 4. Play"""
import numpy as np


def play(env, Q, max_steps=100):
    """Plays an episode using the trained Q-table
    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray shape (state, action) containing the trained Q-table
        max_steps: Maximum number of steps in the episode
    Returns:
        total_rewards: the total rewards for the episode
    """
    total_rewards = 0
    state, _ = env.reset()
    print(env.render())

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, _, _ = env.step(action)
        total_rewards += reward
        print(env.render())
        if terminated:
            break
    return total_rewards
