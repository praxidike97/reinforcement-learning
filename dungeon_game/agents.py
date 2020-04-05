from abc import ABC, abstractmethod
import random
from Action import Action

import numpy as np

from keras.models import Sequential
from keras.layers import Dense


class Agent(ABC):

    @abstractmethod
    def get_next_action(self, state):
        pass

    @abstractmethod
    def update(self, old_state, new_state, action, reward):
        pass


class RandomAgent(Agent):
    def __init__(self):
        self.q_table = None

    def __str__(self):
        return "RandomAgent"

    def get_next_action(self, state):
        return Action.FORWARD if random.random() < 0.5 else Action.BACKWARD

    def update(self, old_state, new_state, action, reward):
        pass


class AccountantAgent(Agent):
    def __init__(self, length=5):
        #self.q_table = [[0,0,0,0,0], [0,0,0,0,0]]
        self.q_table = [[0 for _ in range(length)] for _ in range(2)]

    def __str__(self):
        return "AccountantAgent"

    def get_next_action(self, state):
        if self.q_table[Action.FORWARD.value][state] > self.q_table[Action.BACKWARD.value][state]:
            return Action.FORWARD
        elif self.q_table[Action.FORWARD.value][state] < self.q_table[Action.BACKWARD.value][state]:
            return Action.BACKWARD
        else:
            return Action.BACKWARD if random.random() < 0.5 else Action.FORWARD

    def update(self, old_state, new_state, action, reward):
        self.q_table[action.value][old_state] += reward


class QLearningAgent(Agent):
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.q_table = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # Spreadsheet (Q-table) for rewards accounting
        self.learning_rate = learning_rate  # How much we appreciate new q-value over current
        self.discount = discount  # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.exploration_delta = exploration_rate / iterations  # Shift from exploration to explotation

    def greedy_action(self, state):
        if self.q_table[Action.FORWARD.value][state] > self.q_table[Action.BACKWARD.value][state]:
            return Action.FORWARD
        elif self.q_table[Action.FORWARD.value][state] < self.q_table[Action.BACKWARD.value][state]:
            return Action.BACKWARD
        else:
            return Action.BACKWARD if random.random() < 0.5 else Action.FORWARD

    def random_action(self):
        return Action.FORWARD if random.random() < 0.5 else Action.BACKWARD

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:
            return self.greedy_action(state)
        else:
            return self.random_action()

    def __str__(self):
        return "QLearningAgent"

    def update(self, old_state, new_state, action, reward):
        # Apply the Q-Learning technique
        old_q_value = self.q_table[action.value][old_state]

        next_action = self.greedy_action(new_state)
        max_next_q_value = self.q_table[next_action.value][new_state]

        new_q_value = old_q_value + self.learning_rate*(reward + self.discount*max_next_q_value - old_q_value)
        self.q_table[action.value][old_state] = new_q_value

        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta


class DeepQLearningAgent(Agent):
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.learning_rate = learning_rate
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate # Initial exploration rate
        self.exploration_delta = exploration_rate / iterations # Shift from exploration to explotation

        # Input has five neurons, each represents single game state (0-4)
        self.input_count = 5
        # Output is two neurons, each represents Q-value for action (FORWARD and BACKWARD)
        self.output_count = 2

        self.model = self.get_model(input_size=5, output_size=2)

    def __str__(self):
        return "DeepQLearningAgent"

    def get_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(units=16, input_shape=(input_size,), activation="sigmoid"))
        model.add(Dense(units=16, activation="sigmoid"))
        model.add(Dense(units=output_size, activation="linear"))

        model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
        return model

    def get_Q_values(self, state):
        return self.model.predict(x=np.array(self.to_one_hot(state)))[0]

    def to_one_hot(self, state):
        one_hot = np.zeros((1, 5))
        one_hot[0, state] = 1
        return one_hot

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        return Action.FORWARD if np.argmax(self.get_Q_values(state)) == 0 else Action.BACKWARD

    def random_action(self):
        return Action.FORWARD if random.random() < 0.5 else Action.BACKWARD

    def train(self, old_state, action, reward, new_state):
        old_state_Q_values = self.get_Q_values(old_state)
        new_state_Q_values = self.get_Q_values(new_state)

        # The target to train on
        old_state_Q_values[action.value] = reward + self.discount * np.amax(new_state_Q_values)

        training_input = self.to_one_hot(old_state)
        target_output = np.array([old_state_Q_values])
        self.model.fit(x=training_input, y=target_output)

    def update(self, old_state, new_state, action, reward):
        # Train our model with new data
        print(action)
        self.train(old_state, action, reward, new_state)

        # Finally shift our exploration_rate toward zero (less gambling)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta