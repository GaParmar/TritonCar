import os, sys, time, pdb

from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from vae import VAE

batch_size=32
num_epochs=50
z_size=64
lr=1e-3

vae = VAE(
    label="lab335_32x16",
    image_W=32,
    image_H=16,
    channel_num=3,
    kernel_num=128,
    z_size=z_size,
)
vae.train()
optimizer = torch.optim.Adam(vae.parameters(), lr=lr,
                    weight_decay=1e-5)
# pdb.set_trace()
# make the dataset
dset = torchvision.datasets.ImageFolder(root="vae_images",
        transform=torchvision.transforms.ToTensor())
loader = DataLoader(dset, batch_size=batch_size, shuffle=True)

# encoded shape torch.Size([1, 128, 2, 4])
# mean torch.Size([1, 32]) logvar torch.Size([1, 32])
# z torch.Size([1, 32])
# z_proj torch.Size([1, 128, 4, 2])
# x recon torch.Size([1, 3, 32, 16])


def visualize_image(tensor, name, label=None, env='main', w=250, h=250,
                    update_window_without_label=False):
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = _vis(env).image(
        tensor.numpy(), win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).image(
            tensor.numpy(), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )

for epoch in range(num_epochs):
    data_stream = tqdm(enumerate(loader, 1))
    for batch_index, (x, _) in data_stream:
        iteration = (epoch)*(len(dset)//batch_size) + batch_index
        optimizer.zero_grad()
        (mean, logvar), x_reconstructed = vae(x)
        reconstruction_loss = vae.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = vae.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss
        total_loss.backward()
        optimizer.step()
        data_stream.set_description((
            f'epoch: {epoch} | '
            f'iteration: {iteration} | '
            f'progress: [{batch_index * len(x)}/{len(dset)}] ({(100. * batch_index / len(loader)):.0f}%) | '
            f'loss => '
            f'total: {total_loss.data.item():.4f} / '
            f're: {reconstruction_loss.data.item():.3f} / '
            f'kl: {kl_divergence_loss.data.item():.3f}'
        ))
        if iteration % 500 == 0:
            images = vae.sample(32)
            torchvision.utils.save_image(images, f"samples/{iteration}.png", nrow=5)
            cmb = torch.zeros((20,3,16,32))
            cmb[0:10,:,:,:] = x[0:10,:,:,:]
            cmb[10:,:,:,:] = x_reconstructed[0:10,:,:,:] 
            torchvision.utils.save_image(cmb, f"recons/{iteration}.png", nrow=10)
    