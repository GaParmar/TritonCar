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

ds_train = CarDataset(root=TRAIN_DATASET_ROOT, W=IMAGE_WIDTH, H=IMAGE_HEIGHT, split="train", stochastic=False)
ds_test = CarDataset(root=TRAIN_DATASET_ROOT, W=IMAGE_WIDTH, H=IMAGE_HEIGHT, split="test", stochastic=False)

output_ch = 2

device = torch.device("cuda")

model = LinearPilot(output_ch=output_ch, stochastic=False).cuda()
opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR,
                    weight_decay=1e-5)

loader_train = DataLoader(ds_train, batch_size=TRAIN_BATCH_SIZE,
                            shuffle=True, pin_memory=True, num_workers=4)
loader_test = DataLoader(ds_test, batch_size=TRAIN_BATCH_SIZE,
                            shuffle=True, pin_memory=True, num_workers=4)

L_train, L_test = [], []

for epoch in range(TRAIN_EPOCHS):
    pbar = tqdm(enumerate(loader_train, 1))
    train_loss = 0.0
    model = model.train()
    for idx, batch in pbar:
        opt.zero_grad()
        # batch["image"] is [B, C, H, W]
        # batch["throttle"].shape == batch["steer"].shape == [32,1]
        img = batch["image"][:,0:3,:,:].to(device)
        pred_throttle, pred_steer = model(img)
        mse_loss = F.mse_loss(pred_throttle.view(-1), batch["throttle"].to(device))
        mse_loss += F.mse_loss(pred_steer.view(-1), batch["steer"].to(device))*LAMBDA_STEER
        mse_loss.backward()
        opt.step()
        train_loss += mse_loss.item()
        pbar.set_description(f"epoch: {epoch:3d}    it:{idx:4d}    train_loss:{mse_loss.item():.2f}\t\t")
    L_train.append(train_loss/len(loader_train))
    test_loss = 0.0
    model = model.eval()
    pbar = tqdm(enumerate(loader_test, 1))
    for idx, batch in pbar:
        with torch.no_grad():
            img = batch["image"][:,0:3,:,:].to(device)
            pred_throttle, pred_steer = model(img)
            mse_loss = F.mse_loss(pred_throttle.view(-1), batch["throttle"].to(device))
            mse_loss += F.mse_loss(pred_steer.view(-1), batch["steer"].to(device))*LAMBDA_STEER
            test_loss += mse_loss.item()
        pbar.set_description(f"epoch: {epoch:3d}    it:{idx:4d}    test_loss: {mse_loss.item():.2f}\t\t")
    L_test.append(test_loss/len(loader_test))
    # save the model to file
    #save_path = os.path.join(TRAIN_SU_outpath, f"M_{TRAIN_SU_EXP_NAME}_{epoch}_testL_{test_loss:.2f}.sd")
    #torch.save(model.state_dict(), save_path)
    print(f"{epoch}:: total train_loss: {train_loss:.2f}    test_loss: {test_loss:.2f}")

# plot the losses
plt.plot(all_train_losses, label="train loss")
plt.plot(all_test_losses, label="test loss")
plt.legend()
save_path = os.path.join(f"losses.png")
plt.savefig(save_path)