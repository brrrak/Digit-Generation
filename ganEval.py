import os
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import trange
from ganTrain import Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--complex', type=int, default=1, help='Whether to use the more complex model or simpler one.')
    parser.add_argument('--wgan', type=int, default=0, help='Whether to use the Wasserstein loss.')
    parser.add_argument('--generateForFID', type=int, default=0, help='Generates and saves 60000 images for FID.')
    opt = parser.parse_args()
    print(opt)

    dataset = datasets.MNIST(root='./data', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=64,
                                        shuffle=True)

    # Number of channels in image
    nc = 1
    # Input noise dimension
    nz = 32

    if opt.complex:
        # Complex GAN
        ngf = 32 # Coefficient for number of generator filters
        if opt.wgan:
            checkpoint = torch.load('./model_checkpoints/complexWGAN.tar')
        else:
            checkpoint = torch.load('./model_checkpoints/complexBCEGAN.tar')
    else:
        # Simple GAN
        ngf = 2
        if opt.wgan:
            checkpoint = torch.load('./model_checkpoints/simpleWGAN.tar')
        else:
            checkpoint = torch.load('./model_checkpoints/simpleBCEGAN.tar')
    
    # Initialize generator
    netG = Generator(ngf=ngf).to(device)
    print(netG)

    netG.load_state_dict(checkpoint['generator_state_dict'])

    netG.eval()

    # Get real batch from dataloader
    real_batch = next(iter(dataloader))

    # Plot original images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Get fake images
    with torch.no_grad():
        noise = torch.randn(64, 32, 1, 1, device=device)
        fake_batch = netG(noise)
    # Plot reconstructed images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.show(block=True)

    # Generating Fake Images with the same noise vector as VAE
    torch.manual_seed(101)
    genz = torch.randn(100, 32)
    genz = genz.reshape((100, 32, 1, 1)).to(device)
    # print(genz[0,:,0,0])
    with torch.no_grad():
        samples = netG(genz)
    
    # Plotting Fake Images
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.title("Generated Images Using Same Noise as VAE")
    plt.imshow(np.transpose(vutils.make_grid(samples.to(device)[:100], padding=5, normalize=True, nrow=10).cpu(),(1,2,0)))
    plt.show(block=True)

    if opt.generateForFID:
        print("Present Working Directory: ",os.getcwd())
        os.makedirs("gen_images", exist_ok=True)
        genPath = ''
        if opt.complex:
            genPath = genPath + 'complex'
        else:
            genPath = genPath + 'simple'
        if opt.wgan:
            genPath = genPath + 'WGAN'
        else:
            genPath = genPath + 'DCGAN'
        
        print(f"Saving generated images to: {os.getcwd()}/gen_images/{genPath}/")

        os.makedirs("gen_images/"+genPath, exist_ok=False)
        from cleanfid import fid
        with trange(600) as pbar:
            for i in pbar:
                noise = torch.randn(100, 32, 1, 1, device=device)
                fake_batch = netG(noise)
                for j in range(100):
                    vutils.save_image(fake_batch[j], "gen_images/"+genPath+"/g"+str(i*100 + j)+".png")

        print("Test stats exist for mnist: ", fid.test_stats_exists("mnist", mode="clean"))
        if not fid.test_stats_exists("mnist", mode="clean"):
            print("Creating test stats for mnist.")
            fid.make_custom_stats("mnist", "./mnist_png/training/", mode="clean")
            print("Test stats created for mnist.")

        print("Calculating FID Score for", genPath)
        fidscore = fid.compute_fid("./gen_images/"+genPath+"/", 
                                dataset_name="mnist", 
                                mode="clean", 
                                dataset_split="custom",
                                num_workers=8)
        print("FID Scores calculated.")
        print("Calculating KID Scores for", genPath)
        kidscore = fid.compute_kid("./gen_images/"+genPath+"/", 
                                dataset_name="mnist", 
                                mode="clean", 
                                dataset_split="custom",
                                num_workers=8)
        print("KID Scores calculated.")

        print(f"FID Score for {genPath}: {fidscore}")
        print(f"KID Score for {genPath}: {kidscore}")
