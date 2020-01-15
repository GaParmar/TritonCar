import os, sys, time, math
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

def segment_img(img, color=(165,125,60), threshold=15, size=(120,160,3)):
    np_img = np.array(img)
    rgb_mask = np.ones(size)*color
    threshold_mask = np.ones(size)*threshold
    f_img = np.abs(np_img-rgb_mask)<=threshold_mask
    f_img = np.logical_and(f_img[:,:,0], f_img[:,:,1], f_img[:,:,2])*255
    return Image.fromarray(f_img.astype('uint8'))

def segment_split_norm(img):
    pil_left = segment_img(img.crop((0,0,320,240)).resize([160,120]))
    pil_right = segment_img(img.crop((320,0,640,240)).resize([160,120]))
    img = np.concatenate((np.array(pil_left).reshape(120,160,1),np.array(pil_right).reshape(120,160,1)), axis=2).astype(np.float32)
    # normalize to range [0,1] from [0,255]
    img /= 255.0
    return img

def gen(path_list, batch_size=10, transform=norm_split):
    while True:
        # shuffle
        random.shuffle(path_list)
        i = 0
        num_batches = math.floor(len(path_list)/batch_size)
        print(f"number of batches in the generator = {num_batches}")
        for i in range(num_batches):
            X = np.zeros((batch_size, 120, 160, 6))
            y = [np.zeros((batch_size, 1)), np.zeros((batch_size, 1))]
            batch_paths = path_list[i:i+batch_size]
            for j in range(batch_size):
                impath = batch_paths[j]
                img = transform(Image.open(impath))
                label_str = os.path.basename(impath).replace(".png","").split("_")
                throttle = np.array(int(float(label_str[-2])))
                steer = np.array(int(float(label_str[-1])))
                X[j] = img
                y[0][j] = throttle
                y[1][j] = steer
            yield X,y