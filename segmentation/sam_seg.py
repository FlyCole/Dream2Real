import gc
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from vis_utils import visimg

import os
import pathlib
curr_dir_path = pathlib.Path(__file__).parent.absolute()
working_dir = os.path.join(curr_dir_path, '..')

import pdb


class Segmentor():
    def __init__(self, device="cuda:0"):
        total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if total_memory_gb > 10:
            self.sam = sam_model_registry["vit_h"](checkpoint=os.path.join(working_dir, "models/sam_vit_h_4b8939.pth")).to(device)
        else:
            self.sam = sam_model_registry["vit_b"](checkpoint=os.path.join(working_dir, "models/sam_vit_b_01ec64.pth")).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam,
                                                        points_per_side=48,
                                                        pred_iou_thresh=0.95,
                                                        stability_score_thresh=0.90,
                                                        crop_n_layers=2,
                                                        crop_n_points_downscale_factor=2,
                                                        crop_nms_thresh=0.95,
                                                        min_mask_region_area=120,)
        self.device = device

    def subpart_suppression(self, masks, threshold=0.1):
        # For any pair of objects, if (subpart_threshold) of one is inside the other, keep the other.
        remove_idxs = []
        for i in range(len(masks)):
            curr_mask = masks[i]
            curr_area = curr_mask.sum()
            for j in range(i + 1, len(masks)):
                other_mask = masks[j]
                other_area = other_mask.sum()
                intersection = (curr_mask & other_mask).sum()
                if intersection / curr_area > threshold or intersection / other_area > threshold:
                    # Remove the smaller one.
                    smaller_area_idx = i if curr_area < other_area else j
                    remove_idxs.append(smaller_area_idx)

        keep_idxs = [i for i in range(len(masks)) if i not in remove_idxs]
        masks = [masks[i] for i in keep_idxs]
        return masks

    def large_obj_suppression(self, masks, img, threshold=0.3):
        img_area = img.shape[0] * img.shape[1]
        masks = [mask for mask in masks if mask.sum() / img_area <= threshold]
        return masks

    def small_obj_suppression(self, masks, area_thresh=80, side_thresh=20):
        masks = [mask for mask in masks if mask.sum() >= area_thresh]
        masks = [mask for mask in masks if get_smallest_side(mask)[1] > side_thresh]
        return masks

    # Keeps only masks which are connected components (no multiple islands).
    # Dilate a bit first in case small gap between components but actually same object.
    def disconnected_components_suppression(self, masks, img):
        masks = [mask for mask in masks if cv2.connectedComponents(cv2.dilate(mask.cpu().numpy().astype(np.uint8), np.ones((5, 5), np.uint8)))[0] == 2]
        return masks

    def segment(self, img, show_masks=False, scene_bound_mask=None):
        print("SAM segmenting the image...")
        masks = self.mask_generator.generate(img) # img in HWC uint8 format. Seems like RGB (rather than BGR).

        # if show_masks:
        #     plt.figure(figsize=(20,20))
        #     plt.imshow(img)
        #     show_anns(masks)
        #     plt.axis('off')
        #     plt.show()

        masks = [torch.tensor(mask['segmentation']).to(self.device) for mask in masks]
        if scene_bound_mask is not None:
            scene_bound_mask = torch.tensor(scene_bound_mask.copy()).to(masks[0].device)
            for mask in masks:
                mask &= scene_bound_mask

        # Uncomment for debugging.
        # print(f'Number of masks from SAM after SAM post-proc + before our post-proc: {len(masks)}')
        # if os.path.exists('temp_masks'):
        #     os.system('rm -rf temp_masks')
        # os.mkdir('temp_masks')
        # for i, mask in enumerate(masks):
        #     cv2.imwrite(f'temp_masks/mask_{i:03}.png', mask.cpu().numpy().astype(np.uint8) * 255)

        masks = self.disconnected_components_suppression(masks, img)
        masks = self.large_obj_suppression(masks, img) # To remove bground objs.
        masks = self.subpart_suppression(masks)
        masks = self.small_obj_suppression(masks) # To remove small objs which cannot be grasped anyway.

        # Uncomment for debugging.
        # print(f'Number of masks from SAM after SAM post-proc + our post-proc: {len(masks)}')
        # if os.path.exists('temp_masks'):
        #     os.system('rm -rf temp_masks')
        # os.mkdir('temp_masks')
        # for i, mask in enumerate(masks):
        #     cv2.imwrite(f'temp_masks/mask_{i:03}.png', mask.cpu().numpy().astype(np.uint8) * 255)

        # Inflate object masks to remove shadows on background, which would influence inpainting of holes in background.
        inflation_factor = 1.6
        obj_masks_inflated = [rescale_mask(mask.cpu().numpy(), inflation_factor) for mask in masks]
        obj_masks_inflated = np.logical_or.reduce(obj_masks_inflated)
        obj_masks_inflated = torch.from_numpy(obj_masks_inflated).to(self.device)
        bground_mask = ~obj_masks_inflated
        masks.insert(0, bground_mask)
        print('SAM segmentation complete.')
        return masks

    # Inputs are torch tensors.
    # Output has alpha channel, so has shape (H, W, 4).
    # Returns a tight object image with no background. Used for rendering.
    def get_obj_img(self, img, obj_mask):
        obj_img = img

        row_has_obj = torch.any(obj_mask.view(obj_mask.shape[0], -1), dim=-1)
        rows_with_obj = torch.where(row_has_obj)[0]
        first_obj_row = rows_with_obj[0]
        last_obj_row = rows_with_obj[-1]

        col_has_obj = torch.any(obj_mask.permute(1, 0).reshape(obj_mask.shape[1], -1), dim=-1)
        cols_with_obj = torch.where(col_has_obj)[0]
        first_obj_col = cols_with_obj[0]
        last_obj_col = cols_with_obj[-1]

        obj_img = obj_img[first_obj_row:last_obj_row + 1, first_obj_col:last_obj_col + 1]

        # Add alpha channel.
        obj_img = torch.cat([obj_img, obj_mask.to(obj_img.device)[first_obj_row:last_obj_row + 1, first_obj_col:last_obj_col + 1].unsqueeze(-1) * 255], dim=-1)

        return obj_img

    def free(self):
        # print(f'Memory usage before freeing SAM: {torch.cuda.memory_allocated(0)}')
        self.sam = self.sam.to('cpu')
        del self.sam
        del self.mask_generator
        gc.collect()
        torch.cuda.empty_cache()
        # print(f'Memory usage after freeing SAM: {torch.cuda.memory_allocated(0)}')

# Returns centre in (i, j) coordinates, rather than (x, y).
def centre_of_mass(binary_image):
    moments = cv2.moments(binary_image * 1.0)
    centre = np.array([int(moments["m01"] / moments["m00"]), int(moments["m10"] / moments["m00"])])
    return centre

# OPT: could be faster, since dilate, get_biggest_side and findCountours are done one pixel at a time.
# Maybe computing area and using that to determine when to stop would be faster.
def rescale_mask(mask, scale):
    if scale == 1.0:
        return mask

    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    _, length = get_biggest_side(mask)
    new_length = length * scale
    if scale >= 1:
        while length < new_length:
            mask = cv2.dilate(mask, kernel, iterations=1)
            prev_length = length
            _, length = get_biggest_side(mask)
            if prev_length == length:
                return mask
            prev_length = length
    else:
        while length > new_length:
            mask = cv2.erode(mask, kernel, iterations=1)
            prev_length = length
            _, length = get_biggest_side(mask)
            if prev_length == length:
                return mask
            prev_length = length
    return mask

def get_biggest_side(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    mask_im = mask.copy() * 255
    contours, hierarchy = cv2.findContours(mask_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    box_width = np.linalg.norm(box[0] - box[1])
    box_height = np.linalg.norm(box[1] - box[2])

    if box_width > box_height:
        return (box[2] - box[0]) / box_width, box_width
    else:
        return (box[3] - box[1]) / box_height, box_height

def get_smallest_side(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    mask_im = mask.copy() * 255
    contours, hierarchy = cv2.findContours(mask_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    box_width = np.linalg.norm(box[0] - box[1])
    box_height = np.linalg.norm(box[1] - box[2])

    if box_width < box_height:
        return (box[2] - box[0]) / box_width, box_width
    else:
        return (box[3] - box[1]) / box_height, box_height

# From the Segment Anything documentation.
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# E.g. if bound is 0.5, then object bbox must not be outside the centre square of the image (of width 0.5 * img_width).
# Now batched!
def is_within_bounds_2d(poses, full_img_size, obj_img, bound):
    assert obj_img.shape[2] <= 4, f'Object image has shape {obj_img.shape}, but must have shape (H, W, C)'
    obj_width = (obj_img.shape[1] / full_img_size[1]) * 2
    obj_height = (obj_img.shape[0] / full_img_size[0]) * 2
    obj_centres = poses[:, :2] # pose may later contain orientation.
    obj_lefts = obj_centres[:, 0] - obj_width / 2
    obj_rights = obj_centres[:, 0] + obj_width / 2
    obj_tops = obj_centres[:, 1] + obj_height / 2
    obj_bottoms = obj_centres[:, 1] - obj_height / 2
    return (obj_lefts > (-1 + bound)).logical_and(obj_rights < (1 - bound)).logical_and(obj_tops < (1 - bound)).logical_and(obj_bottoms > (-1 + bound))

# Returns a crop with some background (for context) and no alpha channel. Used for captioning.
def get_thumbnail(img, obj_mask, padding=5, use_mask=True):
    if use_mask:
        img = img.clone()
        img[~obj_mask] = 255

    row_has_obj = torch.any(obj_mask.view(obj_mask.shape[0], -1), dim=-1)
    rows_with_obj = torch.where(row_has_obj)[0]
    first_obj_row = rows_with_obj[0]
    last_obj_row = rows_with_obj[-1]

    col_has_obj = torch.any(obj_mask.permute(1, 0).reshape(obj_mask.shape[1], -1), dim=-1)
    cols_with_obj = torch.where(col_has_obj)[0]
    first_obj_col = cols_with_obj[0]
    last_obj_col = cols_with_obj[-1]

    first_row = max(0, first_obj_row - padding)
    last_row = min(img.shape[0] - 1, last_obj_row + padding)
    first_col = max(0, first_obj_col - padding)
    last_col = min(img.shape[1] - 1, last_obj_col + padding)

    thumbnail = img[first_row:last_row + 1, first_col:last_col + 1]
    return thumbnail

# Post-processing for background mask due to seg association issues.
def remove_components_at_edges(mask):
    mask = mask.clone()

    num_comps, comp_img = cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8))
    comp_img = torch.from_numpy(comp_img).to(mask.device)
    for i in range(num_comps):
        comp_mask = comp_img == i
        if mask_touches_edge(comp_mask):
            mask[comp_mask] = 0

    return mask

def mask_touches_edge(mask):
    row_has_obj = torch.any(mask.view(mask.shape[0], -1), dim=-1)
    rows_with_obj = torch.where(row_has_obj)[0]
    first_obj_row = rows_with_obj[0]
    last_obj_row = rows_with_obj[-1]

    col_has_obj = torch.any(mask.permute(1, 0).reshape(mask.shape[1], -1), dim=-1)
    cols_with_obj = torch.where(col_has_obj)[0]
    first_obj_col = cols_with_obj[0]
    last_obj_col = cols_with_obj[-1]

    return first_obj_row == 0 or last_obj_row == mask.shape[0] - 1 or first_obj_col == 0 or last_obj_col == mask.shape[1] - 1

if __name__ == '__main__':
    img = cv2.imread("./data/3d/pool/manual-obj-render.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segmentor = Segmentor()
    segmentor.segment(img, True)
