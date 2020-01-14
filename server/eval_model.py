import os, sys, time, pdb
import random
import numpy as np
from PIL import Image
import keras

from network import KerasLinear

dataset_root = "OUTPUT/lab335_0"

all_file_paths=[]
# get a list of all files in the preprocessed pkl
for file in os.listdir(dataset_root):
    if ".png" in file:
        all_file_paths.append(os.path.join(dataset_root, file))

model = KerasLinear()

model.model.load_weights("training_models/M2/cp-012.hdf5")

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

loss = 0.0
for impath in all_file_paths[0:1000]:
    img = norm_split(Image.open(impath))
    label_str = os.path.basename(impath).replace(".png","").split("_")
    gt_throttle = np.array(float(label_str[-2]))
    gt_steer = np.array(float(label_str[-1]))
    throttle, steer = model.run(img)
    loss += (abs(gt_throttle-throttle))
    loss += (abs(gt_steer-steer))
print(f"loss {loss}")