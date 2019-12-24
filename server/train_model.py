import os, sys, time, pdb
import random
import numpy as np
from PIL import Image
import keras

from dataset import *
from network import *

dataset_root = "OUTPUT/lab335_0"
SPLIT_RATIO=0.85

all_file_paths = []
# get a list of all files in the preprocessed pkl
for file in os.listdir(dataset_root):
    if ".png" in file:
        all_file_paths.append(os.path.join(dataset_root, file))
all_file_paths.sort()
num_train = int(len(all_file_paths)*SPLIT_RATIO)
train_paths = all_file_paths[0:num_train]
val_paths = all_file_paths[num_train:]
            
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