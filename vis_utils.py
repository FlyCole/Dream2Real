import gc
import os
import pdb
import textwrap
import cv2
import numpy as np
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from moviepy import editor as mpy
from tqdm import tqdm

username = os.environ['USER']

pastel_colors = np.array([
    [142, 236, 245],
    [140, 251, 140],
    [255, 120, 120],
    [251, 248, 204],
    [207, 186, 240],
    [241, 192, 232],
    [253, 228, 207],
    [163, 196, 243],
    [144, 219, 244],
    [152, 245, 225],
])

# Takes in an RGB image.
def visimg(img, label="Image", folder_path=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    # Check for depth image or mask.
    if len(img.shape) == 2:
        if img.dtype == np.bool8:
            img = img.astype(np.uint8)
        img = np.stack((img, img, img), axis=0)
        # Set nans to 0.
        img[np.isnan(img)] = 0
        # Renormalise to 0-255.
        img = (img - img.min()) / (img.max() - img.min()) * 255
    # Figure out channel dimension.
    if img.shape[0] in [3, 4]:
        img = img.transpose(1, 2, 0)
    img = img[:, :, :3].astype(np.uint8)
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(folder_path, f'{label}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray(img).show(title=label)
    # cv2.imshow(label, cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0) # For some reason hitting a key exits the whole program now on robot PC?
