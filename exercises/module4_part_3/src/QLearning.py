import operator
from collections import defaultdict

import gym
import numpy as np

env = env = gym.make('Blackjack-v0')
nA = env.action_space.n
epsilon = 0.1
gamma = 1.0
alpha = 0.1


def get_epision_greedy_action_policy(Q, observation):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)

    return A


def qlearning(total_episodes):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for k in range(total_episodes):

        current_state = env.reset()

        while True:

            prob_scores = get_epision_greedy_action_policy(Q, current_state)
            current_action = np.random.choice(np.arange(nA), p=prob_scores)

            next_state, reward, done, _ = env.step(current_action)

            best_next_action = np.argmax(Q[next_state])

            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[current_state][current_action]

            Q[current_state][current_action] = Q[current_state][current_action] + alpha * td_error

            if done:
                break

            current_state = next_state

    return Q


Q = qlearning(50000)

print(Q)

# m = 0
# for k, v in sorted(Q.items()):
#     m += 1
#     print("item ", m, k, v)
#
# for k, v in sorted(Q.items()):
#     print("state: ", k, " action: ", max(v.items(), key=operator.itemgetter(1))[0], " value: ",
#           max(v.items(), key=operator.itemgetter(1))[1])
