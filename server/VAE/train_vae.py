import os, sys, time, pdb

from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

from vae import VAE

## add root to the path
root_path =  os.path.abspath('../..')
if root_path not in sys.path:
    sys.path.append(root_path)

from config import *
from server.dataset import CarDataset

device = torch.device("cuda")

vae = VAE(
    label=VAE_LABEL,
    image_W=VAE_WIDTH,
    image_H=VAE_HEIGHT-CROP_TOP-CROP_BOT,
    channel_num=3,
    kernel_num=128,
    z_size=VAE_ZDIM,
    device=device
).to(device)
vae.train()
optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR,
                    weight_decay=1e-5)

# make the dataset
ds_train = CarDataset(root=TRAIN_DATASET_ROOT, W=VAE_WIDTH, H=VAE_HEIGHT, split="train", stochastic=False)
ds_test = CarDataset(root=TRAIN_DATASET_ROOT, W=VAE_WIDTH, H=VAE_HEIGHT, split="test", stochastic=False)

loader_train = DataLoader(ds_train, batch_size=VAE_BATCH_SIZE,
                            shuffle=True, pin_memory=True, 
                            num_workers=4)
loader_test = DataLoader(ds_test, batch_size=VAE_BATCH_SIZE, 
                            shuffle=True, pin_memory=True, 
                            num_workers=4)

all_train_losses = []
all_test_losses = []

for epoch in range(VAE_EPOCHS):
    data_stream = tqdm(enumerate(loader_train, 1))
    train_epoch_loss, test_epoch_loss = 0.0, 0.0
    vae.train()
    for batch_index, batch in data_stream:
        optimizer.zero_grad()
        x_combined = batch["image"]
        if MODE == "donkey_adapter":
            x_left = x_combined.to(device)
        else:
            x_left = x_combined[:,0:3,:,:].to(device)
        (mean, logvar), x_reconstructed = vae(x_left)
        reconstruction_loss = vae.reconstruction_loss(x_reconstructed, x_left)
        kl_divergence_loss = vae.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss
        total_loss.backward()
        optimizer.step()
        train_epoch_loss+=total_loss.item()
        data_stream.set_description((
            f'epoch: {epoch} | '
            f'iteration: {batch_index} | '
            f'progress: [{batch_index * x_left.shape[0]}/{len(ds_train)}] ({(100. * batch_index / len(loader_train)):.0f}%) | '
            f'loss => '
            f'total: {total_loss.data.item():.4f} / '
            f're: {reconstruction_loss.data.item():.3f} / '
            f'kl: {kl_divergence_loss.data.item():.3f}'
        ))
    print(f"epoch: {epoch} total loss: {train_epoch_loss}")

    data_stream = tqdm(enumerate(loader_test, 1))
    vae.eval()
    test_epoch_loss = 0.0
    for batch_index, batch in data_stream:  
        with torch.no_grad():
            x_left = batch["image"][:,0:3,:,:].to(device)
            _, x_rec = vae(x_left)
            rec_loss = vae.reconstruction_loss(x_rec, x_left)
            test_epoch_loss+=rec_loss.item()

    train_epoch_loss /= len(loader_train)
    test_epoch_loss /= len(loader_test)
    all_train_losses.append(train_epoch_loss)
    all_test_losses.append(test_epoch_loss)


    # make viz
    images = vae.sample(32)
    if not os.path.exists(os.path.join("gen_images", VAE_LABEL)):
        os.makedirs(os.path.join("gen_images", VAE_LABEL))
    outpath = os.path.join("gen_images", VAE_LABEL, f"epoch-{epoch}_train_samples.png")
    torchvision.utils.save_image(images, outpath, nrow=5)
    cmb = torch.zeros((20,3,VAE_HEIGHT, VAE_WIDTH))
    cmb[0:10,:,:,:] = x_left[0:10,:,:,:]
    cmb[10:,:,:,:] = x_reconstructed[0:10,:,:,:]
    outpath = os.path.join("gen_images", VAE_LABEL, f"epoch-{epoch}_train_recons.png")
    torchvision.utils.save_image(cmb, outpath, nrow=10)

    # save the model to file
    if not os.path.exists(VAE_outpath):
        os.makedirs(VAE_outpath)
    save_path = os.path.join(VAE_outpath, f"M_{VAE_LABEL}_{epoch}.sd")
    torch.save(vae.state_dict(), save_path)

# plot the losses
plt.plot(all_train_losses, label="train loss")
plt.plot(all_test_losses, label="test loss")
plt.legend()
save_path = os.path.join(VAE_outpath, f"losses_{VAE_LABEL}.png")
plt.savefig(save_path)