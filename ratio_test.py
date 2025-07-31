from compressed_sensing import rescale_ratio

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

DEPTH_IMAGE_PATH = "/scratchdata/depth_prompting_nyu/gt/0.png"
EST_IMAGE_PATH = "/scratchdata/depth_prompting_nyu/depthformer/0.png"

# read image in grayscale, then downscale it
Xorig = Image.open(DEPTH_IMAGE_PATH)
Xorig = np.array(Xorig, dtype=float)  # convert to float
Xorig /= 1000  # convert to meters
print('Original image shape: {}'.format(Xorig.shape))
print('Original image max, min: {}, {}'.format(Xorig.max(), Xorig.min()))

Xpred = Image.open(EST_IMAGE_PATH)
Xpred = np.array(Xpred, dtype=float)  # convert to float
Xpred /= 1000  # convert to meters
print(Xpred.shape)
print(Xpred.max(), Xpred.min())

R = 0.5
# Sample some r percent of the pixels
Xsample = Xorig.copy()
mask = np.random.rand(*Xpred.shape) < R
Xsample[~mask] = 0  # Set unselected pixels to 0
print(Xsample.max(), Xsample.min())

new_ratio = rescale_ratio(Xsample, Xpred, ORTHANTWISE_C=0.005, relative_C=True)
print('New ratio image max, min: {}, {}'.format(new_ratio.max(), new_ratio.min()))

final_depth = Xpred * new_ratio
mask = 
final_depth = 
print('Final depth image max, min: {}, {}'.format(final_depth.max(), final_depth.min()))

Image.fromarray(final_depth.astype(np.uint16)).save('final_depth_image.png')
plt.imsave("vis.png", final_depth, cmap='gray')

from metric import evaluateMetrics
evaluateMetrics(Xorig, final_depth)
evaluateMetrics(Xorig, Xpred)