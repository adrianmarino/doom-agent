from collections import deque
from numpy import np


class FrameWindow:
    def __init__(self, frame_shape, size):
        self.size = size
        self.frame_shape = frame_shape
        empty_frames = self.__create_empty_frames(frame_shape, size)
        self.frames = deque(empty_frames, maxlen=self.size)

    def append(self, frame): [self.frames.append(frame) for index in range(self.size if len(self.frames) else 1)]

    @staticmethod
    def __create_empty_frames(frame_shape, size): return [np.zeros(frame_shape, dtype=np.int) for index in range(size)]
