from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class OmniDataset(MonoDataset):
    """Superclass for different types of UnderWater dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(OmniDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (5760, 2880)
        self.side_map = {"l": 2, "r": 3}

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):
        pass

    def get_depth(self, folder, frame_index, side, do_flip):
        pass


class OmniUnderWater(OmniDataset):
    def __init__(self, *args, **kwargs):
        super(OmniUnderWater, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "C{}_temp".format(self.side_map[side]), f_str)
        return image_path
