from enum import Enum


class Action(Enum):
    FORWARD = 0
    BACKWARD = 1

    @classmethod
    def reverse(cls, action):
        return Action.FORWARD if action == Action.BACKWARD else Action.BACKWARD
