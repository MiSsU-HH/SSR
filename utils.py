# Copyright 2020 TU Darmstadt
# Licnese: Apache 2.0 License.
# https://github.com/visinf/1-stage-wseg/blob/master/models/mods/pamr.py
import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial
import cv2
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
from scipy.ndimage import binary_fill_holes
import numpy as np
from PIL import Image
import os
#
# Helper modules
#
class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1

        self.weight_check = weight.clone()

        return weight

    def forward(self, x):

        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B,K,H,W = x.size()
        x = x.view(B*K,1,H,W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d]*4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B,K,-1,H,W)

class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

class LocalStDev(LocalAffinity):

    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        # returns (B,K,P,H,W), where P is the number
        # of locations
        x = super(LocalStDev, self).forward(x)

        return x.std(2, keepdim=True)

class LocalAffinityAbs(LocalAffinity):

    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)

#
# PAMR module
#
class PAMR(nn.Module):

    def __init__(self, num_iter=1, dilations=[1]):
        super(PAMR, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)

        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B,K,H,W = x.size()
        _,C,_,_ = mask.size()

        x_std = self.aff_std(x)

        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)
        x = x.mean(1, keepdim=True)
        x = F.softmax(x, 2)

        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW]
            mask = (m * x).sum(2)

        # xvals: [BxCxHxW]
        return mask


def mask2chw(arr):
  # Find the row and column indices where the array is 1
  rows, cols = np.where(arr == 1)
  # Calculate center of the mask
  center_y = int(np.mean(rows))
  center_x = int(np.mean(cols))
  # Calculate height and width of the mask
  height = rows.max() - rows.min() + 1
  width = cols.max() - cols.min() + 1
  return (center_y, center_x), height, width


def apply_visual_prompts(
    image_array,
    mask,
    visual_prompt_type=('circle'),
    visualize=False,
    color=(255, 0, 0),
    thickness=1,
    blur_strength=(15, 15),
):
    """Applies visual prompts to the image."""
    prompted_image = image_array.copy()
    if 'blur' in visual_prompt_type:
        # blur the part out side the mask
        # Blur the entire image
        blurred = cv2.GaussianBlur(prompted_image.copy(), blur_strength, 0)
        # Get the sharp region using the mask
        sharp_region = cv2.bitwise_and(
            prompted_image.copy(),
            prompted_image.copy(),
            mask=np.clip(mask, 0, 255).astype(np.uint8),
        ).astype(np.uint8)
        # Get the blurred region using the inverted mask
        inv_mask = 1 - mask
        blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
        # Combine the sharp and blurred regions
        prompted_image = cv2.add(sharp_region, blurred_region)
    if 'gray' in visual_prompt_type:
        gray = cv2.cvtColor(prompted_image.copy(), cv2.COLOR_BGR2GRAY)
        # make gray part 3 channel
        gray = np.stack([gray, gray, gray], axis=-1)
        # Get the sharp region using the mask
        color_region = cv2.bitwise_and(
            prompted_image.copy(),
            prompted_image.copy(),
            mask=np.clip(mask, 0, 255).astype(np.uint8),
        )
        # Get the blurred region using the inverted mask
        inv_mask = 1 - mask
        gray_region = (gray * inv_mask[:, :, None]).astype(np.uint8)
        # Combine the sharp and blurred regions
        prompted_image = cv2.add(color_region, gray_region)
    if 'black' in visual_prompt_type:
        prompted_image = cv2.bitwise_and(
            prompted_image.copy(),
            prompted_image.copy(),
            mask=np.clip(mask, 0, 255).astype(np.uint8),
        )
    if 'circle' in visual_prompt_type:
        mask_center, mask_height, mask_width = mask2chw(mask)
        center_coordinates = (mask_center[1], mask_center[0])
        axes_length = (mask_width // 2, mask_height // 2)
        prompted_image = cv2.ellipse(
            prompted_image,
            center_coordinates,
            axes_length,
            0,
            0,
            360,
            color,
            thickness,
        )
    if 'rectangle' in visual_prompt_type:
        mask_center, mask_height, mask_width = mask2chw(mask)
        # center_coordinates = (mask_center[1], mask_center[0])
        # axes_length = (mask_width // 2, mask_height // 2)
        start_point = (
            mask_center[1] - mask_width // 2,
            mask_center[0] - mask_height // 2,
        )
        end_point = (
            mask_center[1] + mask_width // 2,
            mask_center[0] + mask_height // 2,
        )
        prompted_image = cv2.rectangle(
            prompted_image, start_point, end_point, color, thickness
        )
    if 'contour' in visual_prompt_type:
        # Find the contours of the mask
        # fill holes for the mask
        mask = binary_fill_holes(mask)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # Draw the contours on the image
        prompted_image = cv2.drawContours(
            prompted_image.copy(), contours, -1, color, thickness
        )

    if visualize:
        cv2.imwrite(os.path.join('masked_img.png'), prompted_image)
    prompted_image = Image.fromarray(prompted_image.astype(np.uint8))
    return prompted_image


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def affinity_propagation(logits, aff=None, mask=None):
    b, n, c = logits.shape
    n_pow = 2
    n_log_iter = 1

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    logits_rw = logits.clone()

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-4)

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _logits = logits[i].reshape(-1, c)
        _aff = aff[i]
        _logits_rw = torch.matmul(_aff, _logits)
        logits_rw[i] = _logits_rw.reshape(logits_rw[i].shape)


    return logits_rw