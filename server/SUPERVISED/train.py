import os, sys, time, pdb
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

## add root to the path
root_path =  os.path.abspath('../..')
if root_path not in sys.path:
    sys.path.append(root_path)

from config import *
from server.dataset import CarDataset
from server.models.pilots import *
from server.VAE.vae import *

ds_train = CarDataset(root=TRAIN_DS_ROOT, W=IMAGE_WIDTH, H=IMAGE_HEIGHT, split="train", stochastic=False)
ds_test = CarDataset(root=TRAIN_DS_ROOT, W=IMAGE_WIDTH, H=IMAGE_HEIGHT, split="test", stochastic=False)

if CAR_FIX_THROTTLE == -1:
    output_ch = 2
else:
    output_ch = 1

device = torch.device("cpu")

# model = LinearPilot(output_ch=output_ch, stochastic=False).cuda()
vae = VAE(label=VAE_LABEL,image_W=VAE_WIDTH,image_H=VAE_HEIGHT,
            channel_num=3,kernel_num=128,z_size=VAE_ZDIM, device=device).to(device)
VAE_PATH = "../VAE/output_models/M_lab335_z32_1.sd"
vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
vae.eval()
model = EncoderPilot(vae, VAE_ZDIM).to(device)
params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc_out.parameters())
opt = torch.optim.Adam(params, lr=TRAIN_LR,
                    weight_decay=1e-5)

loader_train = DataLoader(ds_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

for epoch in range(TRAIN_EPOCHS):
    pbar = tqdm(enumerate(loader_train, 1))
    train_loss = 0.0
    model = model.train()
    for idx, batch in pbar:
        opt.zero_grad()
        # batch["image"] is [B, C, H, W]
        # batch["throttle"].shape == batch["steer"].shape == [32,1]
        if output_ch == 2:
            pred_throttle, pred_steer = model(batch["image"])
            mse_loss = F.mse_loss(pred_throttle, batch["throttle"])
            mse_loss += F.mse_loss(pred_steer, batch["steer"])*LAMBDA_STEER
        else:
            img = batch["image"][:,0:3,:,:] # only use left image
            pred_steer = model(img.to(device))
            mse_loss = F.mse_loss(pred_steer, batch["steer"].view(pred_steer.shape).to(device))*LAMBDA_STEER
        mse_loss.backward()
        opt.step()
        train_loss += (mse_loss.item() / len(ds_train))
        pbar.set_description(f"epoch: {epoch:3d}    it:{idx:4d}    train_loss:{mse_loss.item():.2f}\t\t")
    test_loss = 0.0
    model = model.eval()
    pbar = tqdm(enumerate(loader_test, 1))
    for idx, batch in pbar:
        with torch.no_grad():
            if output_ch==2:
                pred_throttle, pred_steer = model(batch["image"])
                mse_loss = F.mse_loss(pred_throttle, batch["throttle"])
                mse_loss += F.mse_loss(pred_steer, batch["steer"])*LAMBDA_STEER
            else:
                img = batch["image"][:,0:3,:,:] # only use left image
                pred_steer = model(img.to(device))
                mse_loss = F.mse_loss(pred_steer, batch["steer"].view(pred_steer.shape).to(device))*LAMBDA_STEER
            test_loss += (mse_loss.item() / len(ds_train))
        pbar.set_description(f"epoch: {epoch:3d}    it:{idx:4d}    test_loss: {mse_loss.item():.2f}\t\t")
    # save the model to file
    save_path = os.path.join(TRAIN_SU_outpath, f"M_{TRAIN_SU_EXP_NAME}_{epoch}_testL_{test_loss:.2f}.sd")
    torch.save(model.state_dict(), save_path)
    print(f"{epoch}:: mean train_loss: {train_loss:.2f}    test_loss: {test_loss:.2f}")