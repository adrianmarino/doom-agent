import numpy as np

import lib.util.image_utils as iu


class ImagePreProcessor:
    def __init__(self, width, height, chop_bottom_height):
        self.__width = width
        self.__height = height
        self.__chop_bottom_height = chop_bottom_height

    def pre_process(self, img):
        img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)

        if self.__chop_bottom_height > 0:
            img = iu.vertical_crop_image(img, 0, self.__chop_bottom_height)
            # iu.show_img_array(img)

        img = iu.resize_image(img, self.__width, self.__height)
        # iu.show_img_array(img)

        img = iu.rgb_to_gray(img)
        # iu.show_img_array(img)
        return img
