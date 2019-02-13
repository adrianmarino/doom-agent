import numpy as np


class FrameWindowToModelInputConverter:
    def convert(self, frames):
        input_array = np.stack((frames), axis=2)
        input_array = np.expand_dims(input_array, axis=0)
        return input_array
