# https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4

import numpy as np
from collections import defaultdict
from windy_gridworld import WindyGridworldEnv

env = WindyGridworldEnv()
nA = env.action_space.n
epsilon = 0.1
gamma = 1.0
alpha = 0.1


def get_epision_greedy_action_policy(Q, observation):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)

    return A


def sarsa(total_episodes):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for k in range(total_episodes):

        current_state = env.reset()
        prob_scores = get_epision_greedy_action_policy(Q, current_state)
        current_action = np.random.choice(np.arange(nA), p=prob_scores)

        while True:
            next_state, reward, done, _ = env.step(current_action)

            prob_scores_next_state = get_epision_greedy_action_policy(Q, next_state)
            next_action = np.random.choice(np.arange(nA), p=prob_scores_next_state)

            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[current_state][current_action]
            Q[current_state][current_action] = Q[current_state][current_action] + alpha * td_error

            if done:
                break

            current_state = next_state
            current_action = next_action
    return Q


Q = sarsa(100)

print(Q)
