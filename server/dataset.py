import os, sys, time, math, pdb, json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

from config import *

# before split - 640x240
# after split - 320x240
# resize - 160x120
# crop the top height (remove top 40 pixels)
# final - 160x80
def norm_split(img, W, H):
    if MODE == "donkey_adapter":
        w,h = img.size
        # crop off bottom and top 
        # img = img.crop((0, CROP_TOP, w,h-CROP_BOT))
        img = img.resize([W,H])
        img_t = transforms.ToTensor()(img)
        return img_t

    else:
        # crop off the top 1/4 (60 pixels) of the image
        img = img.crop((0,60,640,240))
        w,h = img.size
        left_pil = img.crop((0,0,int(w/2),h)).resize([W,H])
        right_pil = img.crop((int(w/2),0,w,h)).resize([W,H])
        # ToTensor convers to range [0,1]
        left_t = transforms.ToTensor()(left_pil)
        right_t = transforms.ToTensor()(right_pil)
        cmb = torch.cat((left_t, right_t), dim=0)
        return cmb


def make_label(path):
    if MODE == "triton_car":
        label_str = os.path.basename(path).replace(".png","").split("_")
        throttle = float(label_str[-2])
        steer = float(label_str[-1])
    elif MODE == "donkey_adapter":
        idx = os.path.basename(path).split("_")[0]
        rfile = f"record_{idx}.json"
        path = path.replace(os.path.basename(path), rfile)
        with open(path) as f:
            data = json.load(f)
        throttle = data["user/throttle"]
        steer = data["user/angle"]
    return throttle, steer


def convert_label(val, bins):
    diff = torch.abs(torch.tensor(bins)-torch.ones(val))
    # convert to a probability distribution
    diff_prob = torch.softmax(1.0/diff, 0)
    return diff_prob


class CarDataset(Dataset):
    def __init__(self, root, W, H, split="train", stochastic=False):
        super(CarDataset, self).__init__()
        self.all_files = []
        self.stochastic = stochastic
        self.W = W
        self.H = H
        # make a list of all file
        for f in os.listdir(os.path.join(root)):
            if ".png" in f or ".jpg" in f:
                self.all_files.append(os.path.join(root, f))
        self.all_files.sort()
        # if it is train set use the first 90% of the dataset
        if split == "train":
            self.all_files = self.all_files[0:int(len(self.all_files)*0.9)]
        elif split == "test":
            self.all_files = self.all_files[int(len(self.all_files)*0.9):]
        self.transform_image = norm_split
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        throttle, steer = make_label(self.all_files[idx])
        sample = {  "image"      : Image.open(self.all_files[idx]),
                    "throttle"   : torch.tensor(throttle).float(),
                    "steer"      : torch.tensor(steer).float(),
                    "path"       : self.all_files[idx]}
        sample["image"] = self.transform_image(sample["image"], W=self.W, H=self.H)
        if MODE == "donkey_adapter":
            pass # do nothing, it is already transformed
        elif MODE == "triton_car":
            sample["throttle"] = convert_label(sample["throttle"], PILOT_THROTTLE_BINS) if self.stochastic else sample["throttle"]
            sample["steer"] = convert_label(sample["steer"], PILOT_STEER_BINS) if self.stochastic else sample["steer"]
        return sample


"""
LEGACY CODE

def segment_img(img, color=(165,125,60), threshold=15):
    np_img = np.array(img)
    size = np_img.shape
    rgb_mask = np.ones(size)*color
    threshold_mask = np.ones(size)*threshold
    f_img = np.abs(np_img-rgb_mask)<=threshold_mask
    f_img = np.logical_and(f_img[:,:,0], f_img[:,:,1], f_img[:,:,2])*255
    return Image.fromarray(f_img.astype('uint8'))

def gen(path_list, batch_size=10, transform=norm_split):
    while True:
        # shuffle
        random.shuffle(path_list)
        i = 0
        num_batches = math.floor(len(path_list)/batch_size)
        print(f"number of batches in the generator = {num_batches}")
        for i in range(num_batches):
            X = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CH))
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
"""