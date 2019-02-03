from collections import deque
import numpy as np

class FrameWindow:
    def __init__(self, frame_shape, size):
        self.size = size
        self.frame_shape = frame_shape
        self.reset()

    def reset(self):
        empty_frames = self.__create_empty_frames(self.frame_shape, self.size)
        self.frames = deque(empty_frames, maxlen=self.size)

    def append(self, frame):
        [self.frames.append(frame) for index in range(self.size if len(self.frames) else 1)]
        return self

    @staticmethod
    def __create_empty_frames(frame_shape, size): return [np.zeros(frame_shape, dtype=np.int) for index in range(size)]
