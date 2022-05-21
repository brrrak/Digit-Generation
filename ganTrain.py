import os
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
#from IPython.display import HTML
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, nc=1, nz=32, ngf=32):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 3, bias=False),
            # state size. (nc) x 28 x 28
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=32, wgan=False):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
             # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )
        if wgan == False:
            self.main.add_module("sigmoid", nn.Sigmoid())
    def forward(self, x):

        return self.main(x)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--complex', type=int, default=1, help='Whether to use the more complex model or simpler one.')
    parser.add_argument('--wgan', type=int, default=0, help='Whether to use the Wasserstein loss.')
    parser.add_argument('--visualize', type=int, default=1, help='Whether to visualize training loss.')
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
    # Create a fixed random vector to visualize progress of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    ndf = 32 # Coefficient for number of discriminator filters
    if opt.complex:
        # Complex GAN
        ngf = 32 # Coefficient for number of generator filters
    else:
        ngf = 2
    
    # Initialize generator and discriminator
    netG = Generator(ngf=ngf).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, wgan=opt.wgan).to(device)
    netD.apply(weights_init)

    # Setup optimizers
    lr = 0.0001
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
    
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    num_epochs=opt.epochs

    if opt.wgan:
        weight_clipping_limit = 0.01
        one = torch.FloatTensor([1]).to(device)
        mone = one * -1

        # Wasserstein Training Loop

        print("Starting Wasserstein Training Loop...")

        with trange(num_epochs) as pbar:
            # For each epoch
            for epoch in pbar:
                # For each batch in the dataloader
                for i, data in enumerate(dataloader, 0):

                    ############################
                    # (1) Update D network: maximize mean(D(x)) - mean(D(G(z)))
                    ###########################
                    ## Train with all-real batch
                    netD.zero_grad()
                    # weight clipping
                    for p in netD.parameters():
                        p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)
                    # Format batch
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)
                    # Forward pass real batch through D
                    output = netD(real_cpu).mean(0)
                    # Calculate loss on all-real batch
                    errD_real = output.view(1)
                    # Calculate gradients for D in backward pass
                    errD_real.backward(one)
                    D_x = output.item()

                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    # Generate fake image batch with G
                    fake = netG(noise)
                    # Classify all fake batch with D
                    output = netD(fake.detach()).mean(0)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = output.view(1)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward(mone)
                    D_G_z1 = output.item()
                    # Compute error of D as sum over the fake and the real batches
                    wassD = errD_real - errD_fake
                    # Update D
                    optimizerD.step()

                    ############################
                    # (2) Update G network: minimize - mean(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    # label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = netD(fake).mean(0)
                    # Calculate G's loss based on this output
                    errG = output.view(1)
                    # Calculate gradients for G
                    errG.backward(one)
                    D_G_z2 = output.item()
                    # Update G
                    optimizerG.step()

                    # Output training stats

                    if i % 100 == 0:
                        # Save Losses for plotting later
                        G_losses.append(errG.item())
                        D_losses.append(wassD.item())
                        pbar.set_description(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {wassD.item():.3f}\tLoss_G: {errG.item():.3f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

                # Check how the generator is doing by saving G's output on fixed_noise
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        print("Training Complete.")

    else:
        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Cross Entropy Training Loop

        print("Starting Training Loop...")
        # For each epoch
        with trange(num_epochs) as pbar:
            for epoch in pbar:
                # For each batch in the dataloader
                for i, data in enumerate(dataloader, 0):

                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    ## Train with all-real batch
                    netD.zero_grad()
                    # Format batch
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    # Forward pass real batch through D
                    output = netD(real_cpu).view(-1)
                    # Calculate loss on all-real batch
                    errD_real = criterion(output, label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = output.mean().item()

                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    # Generate fake image batch with G
                    fake = netG(noise)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output = netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = netD(fake).view(-1)
                    # Calculate G's loss based on this output
                    errG = criterion(output, label)
                    # Calculate gradients for G
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    # Update G
                    optimizerG.step()

                    # Output training stats
                    if i % 100 == 0:
                        pbar.set_description('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                              % (epoch, num_epochs, i, len(dataloader),
                                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                        # Save Losses for plotting later
                        G_losses.append(errG.item())
                        D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        print("Training Complete.")

    # Visualize training loss
    plt.figure(figsize=(12,8))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block=True)





    



