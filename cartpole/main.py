import gym
import time
from collections import deque
import random
import argparse
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


class DQNSolver():
    def __init__(self, batch_size=20):
        self.memory = deque(maxlen=1000000)
        self.env = gym.make('CartPole-v1')
        self.exploration_rate = 1.0
        self.exploration_rate_min = 0.01
        self.exploration_rate_decay = 0.995
        self.action_space = [0, 1]
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.model = self.create_model()
        self.batch_size = batch_size

    def create_initial_population(self, size=100, threshold=30.):
        print("Create initial population...")
        total_rewards, samples = list(), list()

        while len(self.memory) < size:
            self.env.reset()
            total_reward = 0
            done = False
            sample = list()
            old_state = None

            while not done:
                # Perform a random action
                action = self.env.action_space.sample()
                state, reward, done, info = self.env.step(action)

                #if not old_state is None and not done:
                if not old_state is None:
                    sample.append((old_state, action, reward, state, done))

                old_state = state
                total_reward += reward

            # Only add the samples to the initial population where the reward exceeds the threshold
            if total_reward > threshold:
                self.memory += sample

            print(total_reward)
            total_rewards.append(total_reward)

        print("Finished creating initial population!")
        return samples

    def create_model(self, input_size=4, output_size=2):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_size,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(output_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def get_next_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for old_state, action, reward, state, done in batch:
            old_state_Q_values = self.model.predict(np.array([old_state]))
            new_state_Q_values = self.model.predict(np.array([state]))

            old_state_Q_update = reward

            if not done:
                old_state_Q_update = reward + self.gamma * np.amax(new_state_Q_values[0])

            old_state_Q_values[0][action] = old_state_Q_update

            training_input = np.array([old_state])
            target_output = np.array(old_state_Q_values)
            self.model.fit(training_input, target_output, verbose=0)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)


def test_model(model):
    env = gym.make('CartPole-v1')
    state = env.reset()
    total_reward = 0

    for t in range(500):
        env.render()
        action = np.argmax(model.predict(np.array([state])))

        state, reward, done, info = env.step(action)
        total_reward += reward

        time.sleep(0.05)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    print("Total reward: %f" % total_reward)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--render_test', action='store_true')
    parser.add_argument('--initial_population', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()

    dqnSolver = DQNSolver()

    if FLAGS.initial_population:
        dqnSolver.create_initial_population(size=1000, threshold=80.)

    run = 0
    top_score = 0

    if FLAGS.train:
        while run < FLAGS.runs:
            run += 1
            done = False

            old_state = dqnSolver.env.reset()
            old_state = np.reshape(old_state, [1, 4])
            step = 0
            while not done:
                action = dqnSolver.get_next_action(old_state)
                state, reward, done, info = dqnSolver.env.step(action)
                reward = reward if not done else -reward
                state = np.reshape(state, [1, 4])
                dqnSolver.memory.append((old_state[0], action, reward, state[0], done))

                old_state = state

                if done:
                    print("Run: " + str(run) + ", exploration: " + str(dqnSolver.exploration_rate) + ", score: " + str(step))
                    if step > top_score:
                        top_score = step
                        dqnSolver.model.save("./best_model-%i.h5" % run)
                    break

                step += 1
                dqnSolver.experience_replay()

    if FLAGS.render_test:
        model = load_model("./best_model.h5")
        test_model(model)
