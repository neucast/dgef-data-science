# https://towardsdatascience.com/q-learning-and-sasar-with-python-3775f86bd178
import numpy as np
import copy

# Environemnts
rewards = np.array([[-float('inf'), -float('inf'), 0, 0, -float('inf'), -float('inf'), -float('inf')],
                    [-float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), 0],
                    [0, -float('inf'), -float('inf'), 0, -float('inf'), 100, -float('inf')],
                    [0, -float('inf'), 0, -float('inf'), 0, -float('inf'), -float('inf')],
                    [-float('inf'), 0, -float('inf'), 0, -float('inf'), 100, 0],
                    [-float('inf'), -float('inf'), 0, -float('inf'), 0, 100, 0],
                    [-float('inf'), 0, -float('inf'), -float('inf'), 0, 100, -float('inf')]])

# print("Rewards:", rewards)
# print("Rewards shape:", rewards.shape)

# Parameters
gamma = 0.8
alpha = 0.01
num_episode = 50000
min_difference = 1e-3
goal_state = 5


def QLearning(rewards, goal_state=None, gamma=0.99, alpha=0.01, num_episode=1000, min_difference=1e-5):
    """
    Run Q-learning loop for num_episode iterations or till difference between Q is below min_difference.
    """
    Q = np.zeros(rewards.shape)
    # #print("Q:", Q)
    # #print("Q shape:", Q.shape)
    all_states = np.arange(len(rewards))
    # #print("all_states", all_states)
    # #print("all_states shape:", all_states.shape)
    for i in range(num_episode):
        # #print("episode", i)
        Q_old = copy.deepcopy(Q)
        # initialize state
        initial_state = np.random.choice(all_states)
        # #print("state:", initial_state)
        action = np.random.choice(np.where(rewards[initial_state] != -float('inf'))[0])
        # #print("action:", action)
        Q[initial_state][action] = Q[initial_state][action] + alpha * (
                rewards[initial_state][action] + gamma * np.max(Q[action]) - Q[initial_state][action])
        # print("Q:", Q)
        # print("rewards[initial_state][action]", rewards[initial_state][action])
        # print("Q[action]", Q[action])
        # print("np.max(Q[action])", np.max(Q[action]))
        # input("Press Enter to continue...")
        cur_state = action

        # loop for each step of episode, until reaching goal state
        while cur_state != goal_state:
            # choose action form states using policy derived from Q
            # print("cur_state", cur_state)
            action = np.random.choice(np.where(rewards[cur_state] != -float('inf'))[0])
            # print("action:", action)
            Q[cur_state][action] = Q[cur_state][action] + alpha * (
                    rewards[cur_state][action] + gamma * np.max(Q[action]) - Q[cur_state][action])
            # print("Q:", Q)
            cur_state = action

        # break the loop if converge
        diff = np.sum(Q - Q_old)
        # print("diff", diff)
        if diff < min_difference:
            # print("diff < min_difference:", diff, "<", min_difference)
            break

        # print("The return value is", np.around(Q / np.max(Q) * 100))
    return np.around(Q / np.max(Q) * 100)


Q = QLearning(rewards, goal_state=goal_state, gamma=gamma, alpha=alpha, num_episode=num_episode,
              min_difference=min_difference)
print("Q: ", Q)
