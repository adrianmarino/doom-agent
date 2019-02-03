import numpy as np


class OnHotVectorFactory:
    @staticmethod
    def create(size, one_index):
        one_hot = np.zeros([size])
        one_hot[one_index] = 1
        one_hot = one_hot.astype(int)
        return one_hot.tolist()
