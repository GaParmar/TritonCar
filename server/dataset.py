import os, sys, time
import random
import numpy as np
from PIL import Image

# before split - 640x240
# after split - 320x240
# resize - 160x120
def norm_split(img):
    left = np.array(img.crop((0,0,320,240)).resize([160,120]))
    right = np.array(img.crop((320,0,640,240)).resize([160,120]))
    img = np.concatenate((left,right), axis=2).astype(np.float32)
    # normalize to range [0,1] from [0,255]
    img /= 255.0
    return img

def gen(path_list, batch_size=10):
    while True:
        # shuffle
        random.shuffle(path_list)
        i = 0
        X = np.zeros((batch_size, 120, 160, 6))
        y = [np.zeros((batch_size, 1)), np.zeros((batch_size, 1))]
        for impath in path_list:
            img = norm_split(Image.open(impath))
            label_str = os.path.basename(impath).replace(".png","").split("_")
            throttle = np.array(float(label_str[-2]))
            steer = np.array(float(label_str[-1]))
            if (i+1)<batch_size:
                X[i] = img
                y[0][i] = throttle
                y[1][i] = steer
            elif (i+1)==batch_size:
                yield X,y 
            else:
                i = 0
            i+=1