import os
import numpy as np
from tensorflow.image import ssim
from tensorflow import get_static_value
import scipy.stats

def res_eval(x, y, mask):
    '''
    Calculate the MAE, RMSE, Pearson Correlation, and SSIM, given two matrices, x and y.
    
    Parameters
    -----
    :param array_like x: 2-D matrix x
    :param array_like y: 2-D matrix y
    :param array_like mask: to mask-out on-sea values
    -----
    :return tuple: floats of MAE, RMSE, Pearson Corr., and SSIM
    '''
    assert x.shape == y.shape
    num_valid_grid_points = mask[mask>0].shape[0]

    diff = abs((y - x)*mask) # pixel-wise difference

    # mean absolute error
    mae = np.sum(diff)/num_valid_grid_points

    # root mean square error
    rmse = np.sqrt(np.sum(np.square(diff))/num_valid_grid_points)

    # Pearson Correlation
    flat_y = y.flatten()
    flat_res = x.flatten()
    corr = scipy.stats.pearsonr(flat_res, flat_y)[0] # drop p_values

    # SSIM
    y_max = np.max(y)
    ssim_value = ssim(np.expand_dims(x, axis=-1),
                      np.expand_dims(y, axis=-1),
                      max_val=y_max)
    ssim_value = get_static_value(ssim_value)

    return mae, rmse, corr, ssim_value