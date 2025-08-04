from compressed_sensing import rescale_ratio, rescale_ratio_proportional

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

N = 654
store = []
store_prop = []
benchmark = []

for i in range(230,N):
    DEPTH_IMAGE_PATH = f"/scratchdata/depth_prompting_nyu/gt/{i}.png"
    EST_IMAGE_PATH = f"/scratchdata/depth_prompting_nyu/depthformer/{i}.png"

    # read image in grayscale, then downscale it
    Xorig = Image.open(DEPTH_IMAGE_PATH)
    Xorig = np.array(Xorig, dtype=float)  # convert to float
    Xorig /= 1000  # convert to meters
    #print('Original image shape: {}'.format(Xorig.shape))
    #print('Original image max, min: {}, {}'.format(Xorig.max(), Xorig.min()))

    Xpred = Image.open(EST_IMAGE_PATH)
    Xpred = np.array(Xpred, dtype=float)  # convert to float
    Xpred /= 1000  # convert to meters
    #print(Xpred.shape)
    #print(Xpred.max(), Xpred.min())
    plt.imsave("ratio_gt.png", Xorig/Xpred, cmap='gray')

    R = 0.1
    # Sample some r percent of the pixels
    Xsample = Xorig.copy()
    mask = np.random.rand(*Xpred.shape) < R
    Xsample[~mask] = 0  # Set unselected pixels to 0
    plt.imsave("sampled.png", Xsample, cmap='gray')
    #print(Xsample.max(), Xsample.min())

    new_ratio = rescale_ratio(Xsample, Xpred, relative_C=0.05)
    print('New ratio image max, min: {}, {}'.format(new_ratio.max(), new_ratio.min()))
    plt.imsave("ratio.png", new_ratio, cmap='gray')

    final_depth = Xpred * new_ratio
    mask = Xsample != 0  # Create a mask for the sampled pixels
    final_depth = (final_depth * ~mask) + (Xsample * mask)  # Combine original and new depth
    print('Final depth image max, min: {}, {}'.format(final_depth.max(), final_depth.min()))

    #Image.fromarray(final_depth.astype(np.uint16)).save('final_depth_image.png')
    #plt.imsave("vis.png", final_depth, cmap='gray')
    
    new_ratio_prop = rescale_ratio_proportional(Xsample, Xpred)
    final_depth_prop = Xpred * new_ratio_prop
    mask_prop = Xsample != 0  # Create a mask for the sampled pixels
    final_depth_prop = (final_depth_prop * ~mask_prop) + (Xsample * mask_prop)  # Combine original and new depth

    from metric import evaluateMetrics
    store.append(evaluateMetrics(Xorig, final_depth))
    store_prop.append(evaluateMetrics(Xorig, final_depth_prop))
    benchmark.append(evaluateMetrics(Xorig, Xpred))
   
    break
   
store = np.array(store)
store_prop = np.array(store_prop)
benchmark = np.array(benchmark) 
#print(store.shape)
#print(benchmark.shape)

print(store.mean(axis=0))
print(store_prop.mean(axis=0))
print(benchmark.mean(axis=0))