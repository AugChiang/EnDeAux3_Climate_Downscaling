
from glob import glob
import random
import math
import os
import numpy as np
import tensorflow as tf

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

def concataux(x, auxarr):
    assert len(x.shape) == 2 # HW,C
    auxarr = auxarr.flatten()
    auxarr = np.expand_dims(auxarr, axis=-1) # HW, C
    x = np.concatenate((x, auxarr), axis=-1)
    return x

class MyDataset():
    def __init__(self, xtrpath, ytrpath, x_max, y_max, size, aux_path,
                 shuffle_size, batch_size, seed=None, split=0.9):
        # self.xtrpath = xtrpath
        # self.ytrpath = ytrpath
        self.xtrpath = sorted(glob(xtrpath + '/*.npy'))
        self.ytrpath = sorted(glob(ytrpath + '/*.npy'))
        self.x_size = size[0]
        self.y_size = size[1]
        self.path_list = list(zip(self.xtrpath, self.ytrpath))
        self.aux_path = aux_path

        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.seed = seed
        self.split = split

        self.max = {"input":np.log1p(x_max),"output":np.log1p(y_max)}
        self.len = len(self.xtrpath)

        self.paths = self.__getdatapath__()
        self.aux_boundary_val = {'u':{'min':-19.8128, 'max':22.79312, 'mean':1.49016},
                                 'v':{'min':-23.51668, 'max':23.240267, 'mean':-0.1382056},
                                 'q700':{'min':96824.2, 'max':103687.2, 'mean':100255.7},
                                 'msl':{'min':0, 'max':0.0146, 'mean':0.0073},
                                 't2m':{'min':-5.2, 'max':32.262, 'mean':13.531},
                                 'lr':{'min':0, 'max':1277.652, 'mean':638.826}}

    def __pathcheck__(self):
        if(len(self.xtrpath) != len(self.ytrpath)):
            print("The number of paths is not the same.")
            return False
        else:
            for pathi in range(len(self.xtrpath)):
                if(self.xtrpath[pathi][-12:] != self.ytrpath[pathi][-12:]):
                    print("The corresponding paths of training and truth are not alike.")
                    return False
            return True

    def __getdatapath__(self):
        '''
        Shuffle and split the path_list into training, validation, and test data paths.
        '''
        if(self.__pathcheck__):
            if(self.seed is not None):
                random.Random(self.seed).shuffle(self.path_list) # shuffle the paths with seed
            train_len = math.floor(len(self.path_list)*self.split)
            test_len = math.floor((len(self.path_list) - train_len)/2)
            # train, val, test
            return self.path_list[:train_len], self.path_list[train_len:train_len+test_len], self.path_list[train_len+test_len:]
        return False

    def __datagen__(self, paths):
        for i in range(len(paths)):
            date = paths[i][0][-12:].decode('utf-8') # yyyymmdd.npy
            # print("Date: ", date) # type:=bytes
            x = np.load(paths[i][0]).flatten()
            y = np.load(paths[i][1]).flatten()

            x = np.log1p(x)
            y = np.log1p(y)

            x = Scale01(arr=x, min=0, max=self.max['input'])
            y = Scale01(arr=y, min=0, max=self.max['output'])
            x = np.expand_dims(x,-1) # HW, 1

            if self.aux_path:
                for auxkey in self.aux_path.keys():
                    auxarr = np.load(os.path.join(self.aux_path[auxkey], auxkey+'_'+date)).flatten()
                    if auxkey in ['u', 'v']:
                        auxarr = ScaleNeg11(arr=auxarr, mean=self.aux_boundary_val[auxkey]['mean'])
                    else:
                       auxarr = Scale01(auxarr, min=self.aux_boundary_val[auxkey]['min'],
                                               max=self.aux_boundary_val[auxkey]['max'])   
                    x = concataux(x, auxarr=auxarr)

            x = np.reshape(x, self.x_size)
            y = np.reshape(y, self.y_size)
            yield x, y

    def train_dataset_gen(self):
        # trp, _, __ = self.__getdatapath__()
        trp = self.paths[0]
        random.shuffle(trp)
        train_dataset = tf.data.Dataset.from_generator(
                self.__datagen__, args = [trp],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        train_dataset = train_dataset.shuffle(self.shuffle_size)
        train_dataset = train_dataset.batch(self.batch_size) # 4D
        return train_dataset

    def val_dataset_gen(self):
        # _, val, __ = self.__getdatapath__()
        val = self.paths[1]
        val_dataset = tf.data.Dataset.from_generator(
                self.__datagen__, args = [val],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        val_dataset = val_dataset.batch(self.batch_size)
        return val_dataset
    
    def test_dataset_gen(self):
        # _, __, test = self.__getdatapath__()
        test = self.paths[2]
        test_dataset = tf.data.Dataset.from_generator(
                self.__datagen__, args = [test],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        test_dataset = test_dataset.batch(self.batch_size)
        return test_dataset
    
def datagen(paths, aux_path:dict, x_max, x_size:tuple):
    aux_boundary_val = {'u':{'min':-19.8128, 'max':22.79312, 'mean':1.49016},
                        'v':{'min':-23.51668, 'max':23.240267, 'mean':-0.1382056},
                        'q700':{'min':96824.2, 'max':103687.2, 'mean':100255.7},
                        'msl':{'min':0, 'max':0.0146, 'mean':0.0073},
                        't2m':{'min':-5.2, 'max':32.262, 'mean':13.531},
                        'lr':{'min':0, 'max':1277.652, 'mean':638.826}}
    for i in range(len(paths)):
        date = paths[i][-12:] # yyyymmdd.npy
        # print("Date: ", date) # type:=bytes
        x = np.load(paths[i]).flatten()
        x = np.log1p(x)
        x = Scale01(arr=x, min=0, max=np.log1p(x_max))
        x = np.expand_dims(x,-1)

        if aux_path:
            for auxkey in aux_path.keys():
                auxarr = np.load(os.path.join(aux_path[auxkey], auxkey+'_'+date)).flatten()
                if auxkey in ['u', 'v']:
                    auxarr = ScaleNeg11(arr=auxarr, mean=aux_boundary_val[auxkey]['mean'])
                else:
                    auxarr = Scale01(auxarr, min=aux_boundary_val[auxkey]['min'],
                                            max=aux_boundary_val[auxkey]['max'])   
                x = concataux(x, auxarr=auxarr)

        x = np.reshape(x, x_size)
        yield x, date