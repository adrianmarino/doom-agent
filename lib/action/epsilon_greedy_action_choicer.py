import numpy as np
import random


class EpsilonGreedyActionChoicer:
    def __init__(self, model, action_size, epsilon):
        self.__model = model
        self.__action_size = action_size
        self.__epsilon = epsilon

    def choice_action(self, frames, epsilon):
        if np.random.rand() <= epsilon.value():
            return random.randrange(self.__action_size)

        return self.__model.predict_action_from_frames(frames)

