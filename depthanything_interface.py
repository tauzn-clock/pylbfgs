import os
import sys

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, "Depth-Anything-V2", "metric_depth"))

import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

def get_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitb'
    dataset = 'hypersim'
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'/scratchdata/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
   
    return model

def proportional_rescale(gt, pred, mask):
    scale = np.ones_like(gt)

    gt = gt[mask]
    pred = pred[mask]

    A = np.sum(gt * pred)/ np.sum(pred * pred)
    return A * scale

def rescale_pred(gt, pred):
    mask = gt > 0
    
    scale = proportional_rescale(gt, pred, mask)
    pred *= scale
    return pred


if __name__ == "__main__":
    model = get_model()
    model.to("cuda:0")
    model.eval()

    import cv2
    import matplotlib.pyplot as plt

    raw_img = cv2.imread("/scratchdata/nyu_depth_crop/train/bedroom_0004/rgb_00000.jpg")
    plt.imsave("rgb.png", raw_img)
    print(raw_img.shape)
    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
    print(depth.shape)
    plt.imsave("depth_anything_v2.png", depth, cmap='gray')