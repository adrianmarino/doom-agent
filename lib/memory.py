from collections import deque
import numpy as np


class Memory:
    def __init__(self, size): self.buffer = deque(maxlen=size)

    def add(self, experience): self.buffer.append(experience)

    def shuffle_indexes(self, interval): return np.random.choice(np.arange(self.size()), size=interval, replace=False)

    def size(self): return len(self.buffer)

    def sample(self, batch_size): return [self.buffer[index] for index in self.shuffle_indexes(batch_size)]
