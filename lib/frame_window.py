from collections import deque
from numpy import np


class FrameWindow:
    def __init__(self, frame_shape=(84 ,84), size=4):
        self.size = size
        self.frame_shape = frame_shape
        empty_frames = self.__create_empty_frames(frame_shape, size)
        self.frames = deque(empty_frames, maxlen=self.size)

    @staticmethod
    def __create_empty_frames(frame_shape, size):
        return [np.zeros(frame_shape, dtype=np.int) for index in range(size)]

    def append(self, frame, times): [self.frames.append(frame) for index in range(times)]

    def add(self, frame):
        if len(self.frames):
            self.append(frame, self.size)

            # Stack the frames
            stacked_state = np.stack(self.frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.frames, axis=2)

        return stacked_state, self.frames
