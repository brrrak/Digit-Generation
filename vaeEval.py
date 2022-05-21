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
from tqdm import trange

from vaeTrain import Encoder, Decoder, VAE

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generateForFID', type=int, default=0, help='Generates and saves 60000 images for FID.')
    opt = parser.parse_args()
    print(opt)

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
    # print(genz[0])
    samples = vae.generate(genz)
    
    # Plotting Fake Images
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(samples.to(device)[:100], padding=5, normalize=True, nrow=10).cpu(),(1,2,0)))
    plt.show(block=True)

    if opt.generateForFID:
        print("Present Working Directory: ",os.getcwd())
        os.makedirs("gen_images", exist_ok=True)
        genPath = 'vae'

        print(f"Saving generated images to: {os.getcwd()}/gen_images/{genPath}/")
        os.makedirs("gen_images/"+genPath, exist_ok=False)
        from cleanfid import fid
        with trange(600) as pbar:
            for i in pbar:
                noise = torch.randn((100, 32), device=device)
                fake_batch = vae.generate(noise)
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
