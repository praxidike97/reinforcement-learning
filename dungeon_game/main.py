import argparse

import matplotlib.pyplot as plt
import numpy as np

from dungeon_game import Dungeon
from agents import RandomAgent, AccountantAgent, QLearningAgent, DeepQLearningAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='RANDOM')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--plot', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()

    randomAgent = RandomAgent()
    accountantAgent = AccountantAgent()
    qLearningAgent = QLearningAgent(iterations=FLAGS.iterations)
    deepQLearningAgent = DeepQLearningAgent(iterations=FLAGS.iterations)

    agent_list = [randomAgent, accountantAgent, qLearningAgent, deepQLearningAgent]
    rewards = [list() for _ in range(len(agent_list))]
    dungeon_list = [Dungeon() for _ in range(len(agent_list))]

    for agent_number, agent in enumerate(agent_list):
        for step in range(FLAGS.iterations):
            old_state = dungeon_list[agent_number].state
            action = agent.get_next_action(old_state)
            new_state, reward = dungeon_list[agent_number].perform_action(action)
            agent.update(old_state, new_state, action, reward)

            rewards[agent_number].append(reward)

            if step%100 == 0:
                print("Agent: %s   Step: %i   Reward: %i" % (str(agent), step, sum(rewards[agent_number])))

    if FLAGS.plot:
        plots = list()
        for i, agent in enumerate(agent_list):
            plot, = plt.plot(np.cumsum(rewards[i]))
            plots.append(plot)

        plt.legend(plots, [str(agent) for agent in agent_list])
        plt.xlabel("# Iterations")
        plt.ylabel("Total reward")
        plt.show()
