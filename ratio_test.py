from depthanything_interface import get_model
from compressed_sensing import rescale_ratio

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

ORIG_IMAGE_PATH = "/scratchdata/processed/outdoor_highres_unfiltered/rgb/100.png"
DEPTH_IMAGE_PATH = "/scratchdata/processed/outdoor_highres_unfiltered/depth/100.png"

model = get_model()
model.to("cuda:0")
model.eval()

raw_img = cv2.imread(ORIG_IMAGE_PATH)
Xpred = model.infer_image(raw_img) # HxW depth map in meters in numpy
print(Xpred.shape)
print(Xpred.max(), Xpred.min())

# read image in grayscale, then downscale it
Xorig = Image.open(DEPTH_IMAGE_PATH)
Xorig = np.array(Xorig, dtype=float)  # convert to float
Xorig /= 1000  # convert to meters
print('Original image shape: {}'.format(Xorig.shape))
print('Original image max, min: {}, {}'.format(Xorig.max(), Xorig.min()))

new_ratio = rescale_ratio(Xorig, Xpred)
print('New ratio image max, min: {}, {}'.format(new_ratio.max(), new_ratio.min()))

final_depth = Xpred * new_ratio
print('Final depth image max, min: {}, {}'.format(final_depth.max(), final_depth.min()))

Image.fromarray(final_depth.astype(np.uint16)).save('final_depth_image.png')
plt.imsave("vis.png", final_depth, cmap='gray')

from metric import evaluateMetrics
evaluateMetrics(Xorig, final_depth)