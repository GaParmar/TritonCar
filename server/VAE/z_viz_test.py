import os, sys
import pdb
import time

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np

## add root to the path
root_path =  os.path.abspath('../..')
if root_path not in sys.path:
    sys.path.append(root_path)

from config import *
from vae import VAE
from server.dataset import CarDataset



def load_model(path, zdim):
    device = torch.device("cpu")

    vae = VAE(label=VAE_LABEL,image_W=VAE_WIDTH,image_H=VAE_HEIGHT,
                channel_num=3,kernel_num=128,z_size=zdim,
                device=device).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    return vae

if __name__ == "__main__":
    zdim = 32
    epoch = 49
    model_path = f"output_models_vae/M_lab335_z{zdim}_{epoch}.sd"
    model = load_model(model_path, zdim)
    batch = 10

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    ds_train = CarDataset(root=TRAIN_DS_ROOT, W=VAE_WIDTH, H=VAE_HEIGHT, split="train", stochastic=False)
    ds_test = CarDataset(root=TRAIN_DS_ROOT, W=VAE_WIDTH, H=VAE_HEIGHT, split="test", stochastic=False)
    loader_train = DataLoader(ds_train, batch_size=batch, shuffle=True)
    loader_test = DataLoader(ds_train, batch_size=batch, shuffle=True)
    train_img = next(iter(loader_train))["image"][1:2,0:3,:,:]
    test_img = next(iter(loader_test))["image"][1:2,0:3,:,:]
    (train_mu, _), _ = model(train_img)
    (test_mu, _), _ = model(test_img)

    mods = [-6, -5, -4, -3, 0, 3, 4, 5, 6]
    imgs = torch.zeros((len(mods)*zdim, 3, VAE_HEIGHT, VAE_WIDTH))
    for curr_z in range(zdim):
        for curr_mod in range(len(mods)):
            z_mod = train_mu.clone()
            z_mod[:,curr_z] += mods[curr_mod]*5
            rec_mod = model.only_decode(z_mod)
            imgs[curr_z*len(mods)+curr_mod, :, :, :] = rec_mod[0]
    torchvision.utils.save_image(imgs, "grid.png", nrow=len(mods))
    # pdb.set_trace()
    
    # train_cmb = torch.cat((train_img, train_rec), dim=0)
    # test_cmb = torch.cat((test_img, test_rec), dim=0)
    # torchvision.utils.save_image(train_cmb, "train.png", nrow=batch)
    # torchvision.utils.save_image(test_cmb, "test.png", nrow=batch)