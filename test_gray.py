from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from compressed_sensing import *

IMG_PATH = "/scratchdata/input_image.png"

gray = Image.open(IMG_PATH)  # Convert to grayscale
gray = np.array(gray, dtype=float)  # Convert to float
gray /= gray.max() # Normalize to [0, 1]
print(gray.max(), gray.min())
plt.imsave("gray_image.png", np.array(gray), cmap='gray')

plt.imsave("ori_freq.png", np.log(np.abs(spfft.dctn(gray, norm='ortho'))), cmap='gray')

R = 0.1
sample = gray.copy()
mask = np.random.rand(*gray.shape) < R
print(mask.sum(), mask.shape)
print(sample[mask].mean())
print((sample[mask]**2).mean()**0.5)
#sample[~mask] = np.random.uniform(0,1,sample[~mask].shape)
#sample[~mask] = (sample[mask]**2).mean()**0.5
sample[~mask] = 1  # Set unselected pixels to 0
plt.imsave("sampled_gray.png", sample, cmap='gray')

ri = np.where(mask.T.flatten())[0]
print(len(ri), gray.shape)
print(ri)
b = sample.T.flatten()[ri].astype(float)
print(b, b.shape)
ny, nx = sample.shape
set_global_param(b, (ny, nx), ri)

out = owlqn(nx * ny, evaluate, progress, 0.05)
out = spfft.idctn(out.reshape((nx, ny)).T, norm='ortho') 
print(out.max(), out.min())
plt.imsave("out_gray.png", out, cmap='gray')