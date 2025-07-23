import numpy as np
import torch

def evaluateMetrics(gt, pred):
    mask = gt != 0
    
    gt = gt[mask]
    pred = pred[mask]
    
    ratio = np.maximum(gt / pred, pred / gt)
    delta1 = np.mean(ratio < 1.25)
    print("Delta1: ", delta1)

    absolute_error = np.abs(gt - pred)
    squared_error = absolute_error ** 2
    
    rmse = np.sqrt(np.mean(squared_error))
    mae = np.mean(absolute_error)
    
    print("RMSE: ", rmse)
    print("MAE: ", mae)