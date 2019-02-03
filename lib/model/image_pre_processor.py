import numpy as np
import skimage


class ImagePreProcessor:

    def __init__(self, size): self.__size = size

    def pre_process(self, img):
        img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
        img = skimage.transform.resize(img, self.__size)
        img = skimage.color.rgb2gray(img)
        return img
