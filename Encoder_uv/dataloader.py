
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
            x = np.expand_dims(x,-1)

            if self.aux_path:
                low_reso = np.load(os.path.join(self.aux_path['low_reso'], 'lr_'+date)).flatten()
                u = np.load(os.path.join(self.aux_path['u'], 'u_'+date)).flatten()
                v = np.load(os.path.join(self.aux_path['v'], 'v_'+date)).flatten() 
                q700 = np.load(os.path.join(self.aux_path['q700'], 'q700_'+date)).flatten()
                msl = np.load(os.path.join(self.aux_path['msl'], 'msl_'+date)).flatten()
                t2m = np.load(os.path.join(self.aux_path['t2m'], 't2m_'+date)).flatten()

                low_reso = Scale01(arr=low_reso, max=1277.652)
                u = ScaleNeg11(arr=u, mean=1.49016)
                v = ScaleNeg11(arr=v, mean=-0.1382056)
                q700 = Scale01(q700, min=96824.2, max=103687.2)
                msl = Scale01(msl, min=0, max=0.0146)
                t2m = Scale01(t2m, min=-5.2, max=32.262)

                low_reso = np.expand_dims(low_reso,-1)
                u = np.expand_dims(u,-1)
                v = np.expand_dims(v,-1)
                q700 = np.expand_dims(q700,-1)
                msl = np.expand_dims(msl,-1)
                t2m = np.expand_dims(t2m,-1)
                
                x = np.concatenate((x, low_reso), axis=-1)
                x = np.concatenate((x,u), axis=-1)
                x = np.concatenate((x,v), axis=-1)
                x = np.concatenate((x,q700), axis=-1)
                x = np.concatenate((x,msl), axis=-1)
                x = np.concatenate((x,t2m), axis=-1)

            x = np.reshape(x, self.x_size)
            y = np.reshape(y, self.y_size)
            yield x, y

    def train_dataset_gen(self):
        trp, _, __ = self.__getdatapath__()
        random.shuffle(trp)
        train_dataset = tf.data.Dataset.from_generator(
                self.__datagen__, args = [trp],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        train_dataset = train_dataset.shuffle(self.shuffle_size)
        train_dataset = train_dataset.batch(self.batch_size) # 4D
        return train_dataset

    def val_dataset_gen(self):
        _, val, __ = self.__getdatapath__()
        val_dataset = tf.data.Dataset.from_generator(
                self.__datagen__, args = [val],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        val_dataset = val_dataset.batch(self.batch_size)
        return val_dataset
    
    def test_dataset_gen(self):
        _, __, test = self.__getdatapath__()
        test_dataset = tf.data.Dataset.from_generator(
                self.__datagen__, args = [test],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        test_dataset = test_dataset.batch(self.batch_size)
        return test_dataset