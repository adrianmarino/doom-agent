import numpy as np
import random


class EpsilonGreedyActionChoicer:
    def __init__(self, model, action_size, epsilon=1):
        self.__model = model
        self.__action_size = action_size
        self.__epsilon = epsilon

    def choice_action(self, state):
        if np.random.rand() <= self.__epsilon:
            return random.randrange(self.__action_size)
        else:
            return self.__predict_best_action(state)

    def __predict_best_action(self, state):
        actions_q_values = self.__model.predict(state)
        best_action_number = np.argmax(actions_q_values)
        return best_action_number
