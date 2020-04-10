import gym
import time
from collections import deque
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam, SGD


class DQNSolver():
    def __init__(self, batch_size=20):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('MountainCar-v0')
        self.exploration_rate = 1.0
        self.exploration_rate_min = 0.01
        self.exploration_rate_decay = 0.9999
        self.action_space = [0, 1, 2]
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.model = self.create_model()
        self.batch_size = batch_size

    def create_initial_population(self, size=100, threshold=30.):
        print("Create initial population...")
        total_rewards, samples = list(), list()
        max_heights = list()

        while len(self.memory) < size:
            self.env.reset()
            total_reward = 0
            done = False
            sample = list()
            old_state = None
            max_height = -10

            while not done:
                # Perform a random action
                action = self.env.action_space.sample()
                state, reward, done, info = self.env.step(action)

                if state[0] > max_height:
                    max_height = state[0]

                #if not old_state is None and not done:
                if not old_state is None:
                    sample.append((old_state, action, reward, state, done))

                old_state = state
                total_reward += reward

            print("Max height: %f" % max_height)
            max_heights.append(max_height)

            # Only add the samples to the initial population where the reward exceeds the threshold
            if max_height > threshold:
                self.memory += sample

            print(total_reward)
            total_rewards.append(total_reward)

        print("Finished creating initial population!")
        plt.hist(max_heights)
        plt.show()

        return samples

    def create_model(self, input_size=2, output_size=3):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_size,), activation="relu"))
        model.add(Dense(output_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def get_next_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def experience_replay(self, runs):
        reduction = (1.0 - self.exploration_rate_min) / (runs)
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        training_input, target_output = list(), list()

        for old_state, action, reward, state, done in batch:
            old_state_Q_values = self.model.predict(np.array([old_state]))
            new_state_Q_values = self.model.predict(np.array([state]))

            #old_state_Q_update = reward
            if done and state[0] >= 0.5:
                old_state_Q_values[0][action] = reward #+ 10.
            else:
                old_state_Q_values[0][action] = reward + self.gamma * np.amax(new_state_Q_values[0])

            #old_state_Q_values[0][action] = old_state_Q_update

            #training_input = np.array([old_state])
            #target_output = np.array(old_state_Q_values)
            #self.model.fit(training_input, target_output, verbose=0)
            training_input.append(old_state)
            target_output.append(old_state_Q_values[0])

        self.model.fit(np.asarray(training_input), np.asarray(target_output), verbose=0)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        #if self.exploration_rate > self.exploration_rate_min:
        #    self.exploration_rate -= reduction


def test_model(model):
    env = gym.make('MountainCar-v0')
    state = env.reset()
    total_reward = 0

    for t in range(500):
        env.render()
        action = np.argmax(model.predict(np.array([state])))
        print(action)

        state, reward, done, info = env.step(action)
        total_reward += reward

        #time.sleep(0.05)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    print("Total reward: %f" % total_reward)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--initial_population', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()

    dqnSolver = DQNSolver()

    if FLAGS.initial_population:
        dqnSolver.create_initial_population(size=50000, threshold=-0.25)

    run = 0
    top_score = 0

    if FLAGS.train:
        while run < FLAGS.runs:
            run += 1
            done = False

            old_state = dqnSolver.env.reset()
            old_state = np.reshape(old_state, [1, 2])
            step = 0
            total_reward = 0
            max_height = -10.

            while not done:
                action = dqnSolver.get_next_action(old_state)
                state, reward, done, info = dqnSolver.env.step(action)

                total_reward += reward
                #reward = reward if not done else -reward
                state = np.reshape(state, [1, 2])
                dqnSolver.memory.append((old_state[0], action, reward, state[0], done))

                if state[0][0] > max_height:
                    max_height = state[0][0]

                if run%20 == 0:
                    dqnSolver.env.render()

                old_state = state

                if done:
                    print("Run: " + str(run) + ", exploration: " + str(dqnSolver.exploration_rate) + ", total reward: " + str(total_reward))
                    #if step > top_score:
                        #top_score = step
                    if total_reward > -200.:
                        dqnSolver.model.save("./models/best_model-%i.h5" % run)
                    break

                step += 1

                dqnSolver.experience_replay(runs=FLAGS.runs)
            print("Max height: %f" % max_height)

        #dqnSolver.model.save("./model.h5")

    if FLAGS.test:
        model = load_model("models/best_model-999.h5")
        test_model(model)
