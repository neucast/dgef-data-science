# https://towardsdatascience.com/q-learning-and-sasar-with-python-3775f86bd178
# https://gym.openai.com/
# https://gym.openai.com/envs/FrozenLake-v0/

import numpy as np
import gym

# FrozenLake-v0 gym environment
env = gym.make('FrozenLake-v1')

# Parameters
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.05
gamma = 0.95

# Initializing the Q-value.
Q = np.zeros((env.observation_space.n, env.action_space.n))


# Function to choose the next action with epsilon greedy.
def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    print("action", action)

    return action


# Initializing the reward.
reward = 0

# Starting the SARSA learning.
for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)

    print("state1", state1)
    print("action1", action1)

    while t < max_steps:
        # Visualizing the training.
        env.render()

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        print("state2", state2)
        print("reward", reward)
        print("done", done)
        print("info", info)

        # Choosing the next action.
        action2 = choose_action(state2)
        print("action2", action2)

        # Learning the Q-value.
        Q[state1, action1] = Q[state1, action1] + alpha * (reward + gamma * Q[state2, action2] - Q[state1, action1])

        state1 = state2
        action1 = action2

        print("new state1", state1)
        print("new action2", action1)

        # Updating the respective values.
        t += 1
        reward += 1

        print("new t", t)
        print("new reward", reward)

        # If at the end of learning process.
        if done:
            break

# Evaluating the performance.
print("Performace : ", reward / total_episodes)

# Visualizing the Q-matrix.
print(Q)
