import os, sys, time, pdb, math
import random
import numpy as np
from PIL import Image
import keras

from dataset import *
from network import *

dataset_root = "OUTPUT/lab335_T10"
img_type = "rgb"
model_path = "training_models/335_T10_RGB/cp_{epoch:03d}_{val_loss:.2f}.hdf5"
num_val = 500

all_file_paths = []
# get a list of all files in the preprocessed pkl
for file in os.listdir(dataset_root):
    if ".png" in file:
        all_file_paths.append(os.path.join(dataset_root, file))

random.seed(101)
random.shuffle(all_file_paths)
num_train = len(all_file_paths)-num_val
train_paths = all_file_paths[0:num_train]
val_paths = all_file_paths[num_train:]

train_steps = math.floor(num_train/10.0)

if img_type=="rgb":
    tg = gen(train_paths, transform=norm_split, num_ch=6)
    vg = gen(val_paths, transform=norm_split, num_ch=6)
    model = KerasLinear(input_shape=(120, 160, 6))
elif img_type=="lane_only":
    tg = gen(train_paths, transform=segment_split_norm, num_ch=2)
    vg = gen(val_paths, transform=segment_split_norm, num_ch=2)
    model = KerasLinear(input_shape=(120, 160, 2))
else:
    raise ValueError("type not implemented")

model.train(train_gen=tg, val_gen=vg, 
            train_steps=train_steps,
            saved_model_path=model_path)

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