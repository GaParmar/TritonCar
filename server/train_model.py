import os, sys, time, pdb
import random
import numpy as np
from PIL import Image
import keras

from dataset import DataGenerator
from network import KerasLinear
from keras.models import Sequential

dataset_root = "OUTPUT/lab335_0"
SPLIT_RATIO=0.8

all_file_paths = []
# get a list of all files in the preprocessed pkl
for file in os.listdir(dataset_root):
    if ".png" in file:
        all_file_paths.append(os.path.join(dataset_root, file))
all_file_paths.sort()
num_train = int(len(all_file_paths)*SPLIT_RATIO)
train_paths = all_file_paths[0:num_train]
val_paths = all_file_paths[num_train:]

# before split - 640x240
# after split - 320x240
# resize - 160x120
def norm_split(img):
    left = np.array(img.crop((0,0,320,240)).resize([160,120]))
    right = np.array(img.crop((320,0,640,240)).resize([160,120]))
    img = np.concatenate((left,right), axis=2).astype(np.float32)
    # normalize to range [0,1] from [0,255]
    # img /= 255.0
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
            
tg = gen(train_paths)
vg = gen(val_paths)
model = KerasLinear()

model.train(train_gen=tg, val_gen=vg, saved_model_path="training_models/M2/cp-{epoch:03d}.hdf5")

print("\n\neval the model now")
loss = 0.0
for impath in all_file_paths[0:1000]:
    img = norm_split(Image.open(impath))
    label_str = os.path.basename(impath).replace(".png","").split("_")
    gt_throttle = np.array(float(label_str[-2]))
    gt_steer = np.array(float(label_str[-1]))
    throttle, steer = model.run(img)
    loss += (abs(gt_throttle-throttle))
    loss += (abs(gt_steer-steer))
    # break
print(f"total loss is {loss}")