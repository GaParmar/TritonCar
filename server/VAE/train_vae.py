import os, sys, time, pdb

from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from vae import VAE

## add root to the path
root_path =  os.path.abspath('../..')
if root_path not in sys.path:
    sys.path.append(root_path)

from config import *
from server.dataset import CarDataset

vae = VAE(
    label=VAE_LABEL,
    image_W=VAE_WIDTH,
    image_H=VAE_HEIGHT,
    channel_num=3,
    kernel_num=128,
    z_size=VAE_ZDIM,
)
vae.train()
optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR,
                    weight_decay=1e-5)

# make the dataset
ds_train = CarDataset(root=TRAIN_DS_ROOT, W=VAE_WIDTH, H=VAE_HEIGHT, split="train", stochastic=False)
ds_test = CarDataset(root=TRAIN_DS_ROOT, W=VAE_WIDTH, H=VAE_HEIGHT, split="test", stochastic=False)

loader_train = DataLoader(ds_train, batch_size=VAE_BATCH_SIZE, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=VAE_BATCH_SIZE, shuffle=True)

# encoded shape torch.Size([1, 128, 2, 4])
# mean torch.Size([1, 32]) logvar torch.Size([1, 32])
# z torch.Size([1, 32])
# z_proj torch.Size([1, 128, 4, 2])
# x recon torch.Size([1, 3, 32, 16])

for epoch in range(VAE_EPOCHS):
    data_stream = tqdm(enumerate(loader_train, 1))
    for batch_index, batch in data_stream:
        optimizer.zero_grad()
        x_combined = batch["image"]
        x_left = x_combined[:,0:3,:,:]
        (mean, logvar), x_reconstructed = vae(x_left)
        reconstruction_loss = vae.reconstruction_loss(x_reconstructed, x_left)
        kl_divergence_loss = vae.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss
        total_loss.backward()
        optimizer.step()
        data_stream.set_description((
            f'epoch: {epoch} | '
            f'iteration: {batch_index} | '
            f'progress: [{batch_index * x_left.shape[0]}/{len(ds_train)}] ({(100. * batch_index / len(loader_train)):.0f}%) | '
            f'loss => '
            f'total: {total_loss.data.item():.4f} / '
            f're: {reconstruction_loss.data.item():.3f} / '
            f'kl: {kl_divergence_loss.data.item():.3f}'
        ))
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
    save_path = os.path.join(VAE_outpath, f"M_{VAE_LABEL}_{epoch}.sd")
    torch.save(vae.state_dict(), save_path)