import numpy as np
import json
import os
from Encoder_uv.scaler import Scale01, ScaleNeg11
import random
import math

# read json file
def read_json_as_dict(filename: str)->dict:
    with open(filename) as f_in:
        return json.load(f_in)

def concataux(x, auxarr):
    '''
    Concatenate the 2-D array x with another array auxarr along the axis=-1

    Parameters
    -----
    :param ndarray x: 2-D numpy array
    :param ndarray auxarr: auxiliary array for concatenation
    -----
    :return ndarray: (H,W,C) array
    '''
    assert len(x.shape) == 2 # HW,C
    auxarr = auxarr.flatten()
    auxarr = np.expand_dims(auxarr, axis=-1) # HW, C
    x = np.concatenate((x, auxarr), axis=-1)
    return x

# used for eval
def getdatapath(path_list, split, seed):
    if(seed is not None):
        random.Random(seed).shuffle(path_list) # shuffle the paths with seed
    train_len = math.floor(len(path_list)*split)
    test_len = math.floor((len(path_list) - train_len)/2)
    # train, val, test
    return path_list[:train_len], path_list[train_len:train_len+test_len], path_list[train_len+test_len:]

def datagen(paths,
            aux_path:dict,
            x_max:float,
            x_size:tuple,
            x_use_log1=True):
    '''
    Data generator for predicting data within the given path.

    Parameters
    -----
    :param list paths: data lists
    :param dict aux_path: auxiliary data dict
    :param float x_max: max value of all training data x
    :param tuple x_size: shape of training data 
    :param bool x_use_log1: if use np.log1p(x), defaults to True
    -----
    :yield ndarray, str: data x, date (yyyymmdd)
    '''
    aux_boundary_val = read_json_as_dict(aux_path['stat'])
    
    for i in range(len(paths)):
        date = paths[i][-12:-4] # yyyymmdd
        # print("Date: ", date) # type:=bytes
        x = np.load(paths[i]).flatten()
        if x_use_log1:
            x = np.log1p(x)
        x = Scale01(arr = x,
                    min = 0,
                    max = np.log1p(x_max))
        x = np.expand_dims(x,-1)

        if aux_path:
            for auxkey in aux_path.keys():
                if auxkey == 'stat': # skip the bring-in statistis of aux data.
                    continue
                auxarr = np.load(os.path.join(aux_path[auxkey], (auxkey + '_' + date + '.npy'))).flatten()
                if auxkey in ['u', 'v']:
                    auxarr = ScaleNeg11(arr = auxarr,
                                        mean = aux_boundary_val[auxkey]['mean'])
                else:
                    auxarr = Scale01(auxarr,
                                     min=aux_boundary_val[auxkey]['min'],
                                     max=aux_boundary_val[auxkey]['max'])   
                x = concataux(x, auxarr)

        x = np.reshape(x, x_size)
        yield x, date