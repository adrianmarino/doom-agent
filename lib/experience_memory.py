from collections import deque
import numpy as np


class StateTransitionMemory:
    def __init__(self, size): self.state_transitions = deque(maxlen=size)

    def add(self, check_point): self.state_transitions.append(check_point)

    def shuffle_indexes(self, batch_size): return np.random.choice(np.arange(self.size()), size=batch_size, replace=False)

    def __len__(self): return len(self.state_transitions)

    def samples(self, batch_size): return [self.state_transitions[index] for index in self.shuffle_indexes(batch_size)]
