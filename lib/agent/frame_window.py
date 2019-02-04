from collections import deque
import numpy as np

class FrameWindow:
    def __init__(self, frame_shape, size):
        self.__size = size
        self.__frame_shape = frame_shape
        self.reset()

    def reset(self):
        empty_frames = self.__create_empty_frames(self.__frame_shape, self.__size)
        self.__frames = deque(empty_frames, maxlen=self.__size)

    def append(self, frame):
        [self.__frames.append(frame) for index in range(self.__size if len(self.__frames) else 1)]
        return self

    @staticmethod
    def __create_empty_frames(frame_shape, size): return [np.zeros(frame_shape, dtype=np.int) for index in range(size)]

    def frames(self): return list(self.__frames)
