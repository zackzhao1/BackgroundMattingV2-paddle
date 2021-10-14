#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def gen_trimap(alpha, ksize=3, iterations=10):
    import cv2
    import numpy as np
    alpha = alpha * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


def BatchSAD(pred, target, mask, scale=2):
    # function loss = compute_sad_loss(pred, target, trimap)
    # error_map = (pred - target).abs() / 255.
    # batch_loss = (error_map * mask).view(B, -1).sum(dim=-1)
    # batch_loss = batch_loss / 1000.
    # return batch_loss.data.cpu().numpy()
    B = target.shape[0]
    error_map = (pred - target).abs()
    batch_loss = (error_map.cpu() * (mask == 128)).reshape((B, -1)).sum(axis=-1)
    batch_loss = batch_loss / 1000.
    return batch_loss.sum().item()/B*scale


def BatchMSE(pred, target, mask, scale=2):
    # function loss = compute_mse_loss(pred, target, trimap)
    # error_map = (single(pred) - single(target)) / 255;
    # loss = sum(sum(error_map. ^ 2. * single(trimap == 128))) / sum(sum(single(trimap == 128)));
    B = target.shape[0]
    error_map = (pred - target)
    batch_loss = (error_map.pow(2).cpu() * (mask == 128)).reshape((B, -1)).sum(axis=-1)
    batch_loss = batch_loss / ((mask == 128).astype(float).reshape((B, -1)).sum(axis=-1) + 1.)
    batch_loss = batch_loss * 1000.
    return batch_loss.sum().item()/B*scale
