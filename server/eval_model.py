import os, sys, time, pdb
import random
import numpy as np
from PIL import Image
import keras

from network import KerasLinear
from dataset import *

dataset_root = "OUTPUT/lab335_newmotor_long"

all_file_paths=[]
# get a list of all files in the preprocessed pkl
for file in os.listdir(dataset_root):
    if ".png" in file:
        all_file_paths.append(os.path.join(dataset_root, file))

random.seed(101)
random.shuffle(all_file_paths)
num_val=500
num_train = len(all_file_paths)-num_val
train_paths = all_file_paths[0:num_train]
val_paths = all_file_paths[num_train:]

model = KerasLinear()

model.model.load_weights("training_models/335_newmotor_long/cp-003.hdf5")

loss = 0.0
for impath in val_paths:
    img = norm_split(Image.open(impath))
    label_str = os.path.basename(impath).replace(".png","").split("_")
    gt_throttle = np.array(float(label_str[-2]))
    gt_steer = np.array(float(label_str[-1]))
    throttle, steer = model.run(img)
    loss += (abs(gt_throttle-throttle))
    loss += (abs(gt_steer-steer))
print(f"loss {loss}")