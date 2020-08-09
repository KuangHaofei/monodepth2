# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import networks
from datasets import OmniUnderWater
from layers import disp_to_depth
from utils import readlines


def test_simple():
    """Function to predict for a single image or folder of images
    """
    device = torch.device("cpu")
    model_name = 'weights_19'
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)

    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    file_dir = os.path.dirname(__file__)
    data_path = os.path.join(file_dir, "omni_data")
    split = 'omni_uw'
    fpath = os.path.join(file_dir, "splits", split, "{}_files.txt")
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.jpg'
    frame_ids = [0, -10, 10]
    val_dataset = OmniUnderWater(data_path, val_filenames, feed_height, feed_width,
                                 frame_ids, 4, is_train=False, img_ext=img_ext)
    original_height, original_width = val_dataset.height, val_dataset.width

    batch_size = 1
    val_loader = DataLoader(val_dataset, batch_size, False,
                            num_workers=4, pin_memory=True, drop_last=True)

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for data in val_loader:
            input_color = data[("color", 0, 0)]
            outputs = depth_decoder(encoder(input_color))

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(val_filenames))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())
            print(disp)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            plt.cla()
            plt.imshow(im)
            plt.pause(0.1)

    print('-> Done!')


if __name__ == '__main__':
    test_simple()
