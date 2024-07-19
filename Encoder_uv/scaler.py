import numpy as np

def ScaleNeg11(arr, mean=None):
    '''Min-Max scaler to [-1,1]'''
    if(mean is None):
       mean = np.mean(arr)
    min = np.min(arr)
    max = np.max(arr)
    return (arr-mean)/(max-min)

def Scale01(arr, min=None, max=None):
    '''Min-Max scaler to [0,1]'''
    if(min is None):
        min = np.min(arr)
    if(max is None):
        max = np.max(arr)
    return (arr-min)/(max-min)