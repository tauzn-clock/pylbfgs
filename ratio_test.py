from depth_test import lbfgs_reconstruct_image
from depthanything_interface import get_model

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

ORIG_IMAGE_PATH = "/scratchdata/nyu_depth_crop/train/bedroom_0004/sync_depth_00000.png"

model = get_model()
model.to("cuda:0")
model.eval()

raw_img = cv2.imread("/scratchdata/nyu_depth_crop/train/bedroom_0004/rgb_00000.jpg")
Xpred = model.infer_image(raw_img) # HxW depth map in meters in numpy
Xpred *= 1000  # convert to mm
print(Xpred.shape)
print(Xpred.max(), Xpred.min())

# read image in grayscale, then downscale it
Xorig = Image.open(ORIG_IMAGE_PATH)
Xorig = np.array(Xorig, dtype=float)  # convert to float
print('Original image shape: {}'.format(Xorig.shape))
print('Original image max, min: {}, {}'.format(Xorig.max(), Xorig.min()))

ratio =  Xorig / Xpred
ratio[Xorig == 0] = 0  # avoid division by zero
print('Ratio image max, min: {}, {}'.format(ratio.max(), ratio.min()))
plt.imsave('ratio_image.png', ratio, cmap='gray')

new_ratio = lbfgs_reconstruct_image(ratio)
print('New ratio image max, min: {}, {}'.format(new_ratio.max(), new_ratio.min()))

final_depth = Xpred * new_ratio
print('Final depth image max, min: {}, {}'.format(final_depth.max(), final_depth.min()))

Image.fromarray(final_depth.astype(np.uint16)).save('final_depth_image.png')
plt.imsave("vis.png", final_depth, cmap='gray')

from metric import evaluateMetrics
evaluateMetrics(Xorig, final_depth)