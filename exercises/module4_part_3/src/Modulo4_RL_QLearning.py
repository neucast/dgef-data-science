import gym

import numpy as np
import operator
from collections import defaultdict


class QLearningAgent():
    def __init__(self, possibleActions, learningRate, discountFactor, epsilon, initVals=0.0):
        self.possibleActions = possibleActions
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.state = []
        self.G = initVals
        self.prevState = []
        self.prevAction = None
        self.prevReward = initVals
        self.currState = []
        self.currAction = None
        self.QValues = defaultdict(lambda: {})
        self.policy = defaultdict(float)
        self.numPossActions = len(self.possibleActions)

    def getPolicy(self):
        # Generate e-greedy policy
        if self.state in self.QValues.keys():
            for action in self.possibleActions:
                self.policy[action] = self.epsilon / self.numPossActions
            bestAction = max(self.QValues[self.state].items(), key=operator.itemgetter(1))[0]
            self.policy[bestAction] += (1.0 - self.epsilon)
        else:
            for action in self.possibleActions:
                self.policy[action] = 1 / self.numPossActions

        print("Policy to take action : ")
        for k, v in self.policy.items():
            print("action ", k, "Choose prob ", v)

    def toStateRepresentation(self, state):
        return tuple((state[0], state[1]))

    def setExperience(self, state, action, reward):
        # Save data from t-1 - First step in episode
        if self.prevState == []:
            self.prevState = state
            self.prevAction = action
            self.prevReward = reward

        self.currState = state
        self.currAction = action
        self.currReward = reward

    def setState(self, state):
        self.state = state

    def reset(self):
        self.state = []
        self.prevState = []
        self.prevAction = None
        self.prevReward = 0.0
        self.currState = []
        self.currAction = None
        self.currReward = 0.0

    def act(self):
        # Take an action
        self.getPolicy()
        probs = list(self.policy.values())
        actions = list(self.policy.keys())
        action = actions[np.random.choice(np.arange(len(probs)), p=probs)]
        return action

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def computeHyperparameters(self):
        return self.epsilon

    def learn(self):

        print("-----Learning update-----")

        # Initialize Q[state][action] to zero
        if not self.currState in self.QValues.keys():
            for action in self.possibleActions:
                self.QValues[self.currState][action] = 0

        if not self.prevState in self.QValues.keys():
            for action in self.possibleActions:
                self.QValues[self.prevState][action] = 0

        # If action is none, step is terminal and Q[state][action] is zero.
        if self.currAction != None:
            target = self.prevReward + (self.discountFactor * self.QValues[self.currState][self.currAction])
        else:
            target = self.prevReward

        delta = target - self.QValues[self.prevState][self.prevAction]
        self.QValues[self.prevState][self.prevAction] += self.learningRate * delta

        # Save the current values for next time
        self.prevState = self.currState
        self.prevAction = self.currAction
        self.prevReward = self.currReward

        return self.QValues


if __name__ == '__main__':

    env = gym.make('Blackjack-v0')
    space_size = env.action_space.n
    possibleActions = []
    for i in range(space_size):
        possibleActions.append(str(i))

    # Initialize a QLearning Agent
    agent = QLearningAgent(possibleActions, learningRate=0.1, discountFactor=0.999, epsilon=0.9)

    # Run training QLearning Method
    for episode in range(30):
        print("\n******************************* EPISODE ", episode, "***************************")
        agent.reset()
        observation = env.reset()  # Returns current state
        nextObservation = None
        epsStart = True
        print("State after reset: ", observation)

        done = False

        while done == False:

            epsilon = agent.computeHyperparameters()
            agent.setEpsilon(epsilon)
            obsCopy = observation  # Copy current state
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()

            print("Action: ", action)

            nextObservation, reward, done, status = env.step(int(action))

            best_next_action = np.argmax(agent.QValues[nextObservation])

            print("Next state: ", nextObservation)
            print("Next reward: ", reward)

            print("Episode finished: ", done)
            agent.setExperience(agent.toStateRepresentation(obsCopy), str(best_next_action), reward)

            if not epsStart:
                agent.learn()
            else:
                epsStart = False

            observation = nextObservation

        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None)

        agent.learn()

        QValues = agent.QValues

if __name__ == '__main__':

    env = gym.make('Blackjack-v0')
    space_size = env.action_space.n
    possibleActions = []
    for i in range(space_size):
        possibleActions.append(str(i))

    # Initialize a SARSA Agent
    agent = QLearningAgent(possibleActions, learningRate=0.1, discountFactor=0.999, epsilon=0.9)

    # Run training SARSA Method
    for episode in range(50000):
        agent.reset()
        observation = env.reset()  # Returns current state
        nextObservation = None
        epsStart = True

        done = False

        while done == False:

            epsilon = agent.computeHyperparameters()
            agent.setEpsilon(epsilon)
            obsCopy = observation  # Copy current state
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()

            nextObservation, reward, done, status = env.step(int(action))

            best_next_action = np.argmax(agent.QValues[nextObservation])

            agent.setExperience(agent.toStateRepresentation(obsCopy), str(best_next_action), reward)

            if not epsStart:
                agent.learn()
            else:
                epsStart = False

            observation = nextObservation

        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None)

        agent.learn()

        QValues = agent.QValues

print("Q value function: ")
print(QValues)

# m = 0
# for k, v in sorted(QValues.items()):
#     m += 1
#     print("item ", m, k, v)
#
# for k, v in sorted(QValues.items()):
#     print("state: ", k, " action: ", max(v.items(), key=operator.itemgetter(1))[0], " value: ",
#           max(v.items(), key=operator.itemgetter(1))[1])
#
