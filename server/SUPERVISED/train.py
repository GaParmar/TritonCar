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
from server.models.pilots import LinearPilot

ds_train = CarDataset(root=TRAIN_DS_ROOT, split="train")
ds_test = CarDataset(root=TRAIN_DS_ROOT, split="test")

if CAR_FIX_THROTTLE == -1:
    output_ch = 2
else:
    output_ch = 1

model = LinearPilot(output_ch=output_ch)
opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR,
                    weight_decay=1e-5)

loader_train = DataLoader(ds_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


for epoch in range(TRAIN_EPOCHS):
    pbar = tqdm(enumerate(loader_train, 1))
    train_loss = 0.0
    model = model.train()
    for idx, batch in pbar:
        opt.zero_grad()
        # batch["image"] is [32, 6, 160, 80]
        # batch["throttle"].shape == batch["steer"].shape == [32]
        if output_ch == 2:
        pred_throttle, pred_steer = model(batch["image"])
        mse_loss = F.mse_loss(pred_throttle, batch["throttle"])
        mse_loss += F.mse_loss(pred_steer, batch["steer"])*2.0
        mse_loss.backward()
        opt.step()
        train_loss += (mse_loss.item() / len(ds_train))
        pbar.set_description(f"epoch: {epoch:3d}    it:{idx:4d}    train_loss:{mse_loss.item():.2f}\t\t")
    test_loss = 0.0
    model = model.eval()
    pbar = tqdm(enumerate(loader_test, 1))
    for idx, batch in pbar:
        with torch.no_grad():
            pred_throttle, pred_steer = model(batch["image"])
            mse_loss = F.mse_loss(pred_throttle, batch["throttle"])
            mse_loss += F.mse_loss(pred_steer, batch["steer"])*2.0
            test_loss += (mse_loss.item() / len(ds_train))
        pbar.set_description(f"epoch: {epoch:3d}    it:{idx:4d}    test_loss: {mse_loss.item():.2f}\t\t")
    # save the model to file
    save_path = os.path.join(TRAIN_SU_outpath, f"M_{TRAIN_SU_EXP_NAME}_{epoch}_testL_{test_loss:.2f}")
    torch.save(model.state_dict(), save_path)
    print(f"{epoch}:: mean train_loss: {train_loss:.2f}    test_loss: {test_loss:.2f}")