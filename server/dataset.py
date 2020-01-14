import os, sys, time, pdb
import random
from PIL import Image
import numpy as np
import keras
from tensorflow.python.keras.utils import data_utils

def norm_split(img):
    # normalize to range [0,1] from [0,255]
    img /= 255.0
    # change shape from [240, 640, 3] to [240, 320, 6]
    left = img[:,0:320,:]
    right = img[:,320:,:]
    data = np.concatenate((left, right), axis=2)
    return data

def transform_target(data):
    throttle = data[0]
    steer = data[1]
    # normalize throttle to range [0,1] from [90, 90+15]
    throttle -= 90
    throttle /= 15.0
    # normalize steer to range [0,1] from [60,120]
    steer -= 60
    steer /= 60.0
    return [throttle, steer]

class DataGenerator(data_utils.Sequence):
    'Generates data for Keras'
    """
    all_files - list of files to make the dataset from
    batch_size - number of samples in a single batch
    size - (H,W,C) tuple
    shuffle - whether to shuffle the order of samples
    transform_image - function to apply to the image
    transform_target - preprocessing for target pair
    """
    def __init__(self, all_files, batch_size=10, size=(240,640,3),
                    shuffle=True, transform_image=norm_split, 
                    transform_target=transform_target):
        super(DataGenerator, self).__init__()
        
        self.all_files = all_files
        self.h = size[0]
        self.w = size[1]
        self.c = size[2]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.all_files))
        self.transform_image = transform_image
        self.transform_target = transform_target
        if not self.transform_image:
            self.transform_image = lambda x:x
        if not self.transform_target:
            self.transform_target = lambda x:x
        if self.shuffle:
            random.shuffle(self.all_files)

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.all_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices= self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # list of paths for the current batch
        batch_paths_list = [self.all_files[k] for k in indices]
        # Generate data
        X, y = self.__data_generation(batch_paths_list)
        return X, y


    def __data_generation(self, batch_paths_list):
        'Generates data containing batch_size samples' 
        # X : (n_samples, H, W, C)
        X = np.empty((self.batch_size, self.h, self.w, self.c), dtype=float)
        y = np.empty((self.batch_size, 2), dtype=float)

        # Load data from file
        for i, impath in enumerate(batch_paths_list):
            img = np.array(Image.open(impath)).astype(np.float32)
            label_str = os.path.basename(impath).replace(".png","").split("_")
            throttle = float(label_str[-2])
            steer = float(label_str[-1])
            X[i,] = self.transform_image(img)
            y[i] = self.transform_target([throttle, steer])

        return X, y