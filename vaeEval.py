import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import argparse
import os
import torchvision.utils as vutils

from vaeTrain import Encoder, Decoder, VAE

if __name__ == '__main__':
    dataset = dset.MNIST(root = './data',
                         download = True,
                         transform = transforms.ToTensor())
    batchSize=64
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = VAE(28, 32, 32)
    vae.to(device)
    print(vae)
 
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    checkpoint = torch.load("./model_checkpoints/vae.tar")
    vae.load_state_dict(checkpoint['vae_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    vae.eval()

    # Get real batch from dataloader
    real_batch = next(iter(dataloader))

    # Plot original images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Original Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Get reconstructed images
    with torch.no_grad():
        real_batch = real_batch[0].reshape(-1, 28, 28).to(device)
        _1, _2, fake_batch = vae(real_batch)
    # Plot reconstructed images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Reconstructed Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.show(block=True)

    # Generating Fake Images
    torch.manual_seed(101)
    genz = torch.randn((100, 32))
    print(genz[0])
    samples = vae.generate(genz)
    
    # Plotting Fake Images
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(samples.to(device)[:100], padding=5, normalize=True, nrow=10).cpu(),(1,2,0)))
    plt.show(block=True)
