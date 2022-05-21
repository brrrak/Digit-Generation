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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))

        return out[:, -1, :]

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 16, stride=(1,1), kernel_size=(4,4), padding=0), #4
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, stride=(2,2), kernel_size=(4,4), padding=1), #8
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 4, stride=(2,2), kernel_size=(4,4), padding=1), #16
            nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.ConvTranspose2d(4, 1, stride=(2,2), kernel_size=(4,4), padding=3), #28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return self.main(x)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # LSTM Encoder
        self.encoder = Encoder(input_dim, hidden_dim)
        # Learning Representations
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.fc = nn.Linear(latent_dim, latent_dim)
        # Convolutional Decoder
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        z = z_mu + eps * torch.exp(z_log_var/2)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        x = self.reparameterize(z_mean, z_log_var)
        x = self.fc(x)
        x = self.decoder(x)
        return z_mean, z_log_var, x

    def generate(self, z):
        z = z.to(device)
        z = self.fc(z)
        x = self.decoder(z)
        return x

def loss_function(x_hat, x, mu, logvar):
    # Reconstruction Loss
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # KL Divergence
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return BCE, KLD, BCE + KLD


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--visualize', type=bool, default=0, help='Whether to visualize training loss.')
    parser.add_argument('--save_trained', type=bool, default=0, help='Whether to save the trained model.')
    opt = parser.parse_args()
    print(opt)

    dataset = dset.MNIST(root = './data',
                         download = True,
                         transform = transforms.ToTensor())
    batchSize=64
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=True)
    

    vae = VAE(28, 32, 32)
    vae.to(device)
    print(vae)
    
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    if opt.visualize:
        plt.figure(figsize=(12,8))

    with trange(opt.epochs) as pbar:
        for epoch in pbar:
            epoch_loss = 0.0
            loss_plot_BCE = 0.0
            loss_plot_KLD = 0.0

            for i, data in enumerate(dataloader, 0):
                inputs = data[0].to(device).reshape(-1, 28, 28)

                optimizer.zero_grad()
                z_mean, z_log_var, x = vae(inputs)
                x = x.reshape(-1, 28, 28)

                BCE, KLD, loss = loss_function(x, inputs, z_mean, z_log_var)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                loss_plot_BCE += BCE.item()
                loss_plot_KLD += KLD.item()

            pbar.set_description(f"Epoch {epoch+1} Loss: {epoch_loss:.2f}")

            if opt.visualize:
                plt.scatter(epoch,loss_plot_BCE*len(dataset)/64, color = 'green', label='Reconstruction Term')
                plt.scatter(epoch,loss_plot_KLD*len(dataset)/64, color = 'blue', label='KL Term')
                plt.scatter(epoch,epoch_loss*len(dataset)/64, color = 'red', label='Total Loss')
    if opt.visualize:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show(block=True)

    print('Training Complete.')

    if opt.save_trained:
#        torch.save({
#            'vae_state_dict': vae.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            }, "./model_checkpoints/vae.tar")
        print('Model saved.')




