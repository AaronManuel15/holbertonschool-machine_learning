"""Task 3. Q-learning"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning:
    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containging the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value for updating epsiolon between episodes
    Return:
        Q: the updated Q-table
        total_rewards: list containing the rewards per episode"""

    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        rewards_current_episode = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _, _ = env.step(action)

            if reward == 1 and done:
                rewards_current_episode += reward
            elif reward == 0 and done:
                rewards_current_episode -= 1
                reward = -1

            Q[state, action] = (1 - alpha) * Q[state, action] + \
                alpha * (reward + gamma * np.max(Q[new_state]))
            state = new_state

            if done:
                break
        epsilon = (1 - min_epsilon) * np.exp(-epsilon_decay * episode) +\
            min_epsilon
        total_rewards.append(rewards_current_episode)

    return Q, total_rewards


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
        return np.random.randint(4)
    else:
        return np.argmax(Q[state])
