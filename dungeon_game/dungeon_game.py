import random

from Action import Action


class Dungeon():
    def __init__(self, length=5, slip=0.0, reward_left=2, reward_right=10):
        self.length = length
        self.slip = slip
        self.reward_left = reward_left
        self.reward_right = reward_right
        self.state = 0

    def perform_action(self, action):
        if random.random() < self.slip:
            action = Action.reverse(action)

        reward = 0

        if action == Action.FORWARD:
            self.state = min(self.state+1, self.length-1)
            reward = self.reward_right if self.state == self.length-1 else 0
        elif action == Action.BACKWARD:
            self.state = 0
            reward = self.reward_left

        return self.state, reward

    def reset(self):
        self.state = 0
