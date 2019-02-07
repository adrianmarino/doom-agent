import numpy as np

import lib.util.image_utils as iu


class ImagePreProcessor:
    def __init__(self, size, panel_height=80):
        self.__size = size
        self.__panel_height = panel_height

    def pre_process(self, img):
        img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)

        img = iu.vertical_crop_image(img, 0, self.__panel_height)
        # iu.show_img_array(img)

        img = iu.resize_image(img, self.__size[0], self.__size[1])
        # iu.show_img_array(img)

        img = iu.rgb_to_gray(img)
        # iu.show_img_array(img)
        return img
