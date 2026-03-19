import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import random

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h2 = h2.view(-1, 64 * 7 * 7)
        h3 = F.relu(self.fc1(h2))
        return self.fc21(h3), self.fc22(h3)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        h4 = h4.view(-1, 64, 7, 7)
        h5 = F.relu(self.deconv1(h4))
        return torch.sigmoid(self.deconv2(h5))

def interpolate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    model = ConvVAE().to(device)
    if os.path.exists('vae_conv_mnist.pth'):
        model.load_state_dict(torch.load('vae_conv_mnist.pth', map_location=device))
        model.eval()
        print("Model loaded from vae_conv_mnist.pth")
    else:
        print("Error: vae_conv_mnist.pth not found.")
        return

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=10000, shuffle=False)
    
    data, targets = next(iter(test_loader))
    all_interpolations = []
    
    for digit in range(10):
        indices = (targets == digit).nonzero(as_tuple=True)[0]
        idx1, idx2 = random.sample(indices.tolist(), 2)
        x1, x2 = data[idx1].unsqueeze(0).to(device), data[idx2].unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu1, _ = model.encode(x1)
            mu2, _ = model.encode(x2)
            row_images = [x1]
            for i in range(1, 10):
                lam = i / 10.0
                z_lam = (1 - lam) * mu1 + lam * mu2
                recon = model.decode(z_lam)
                row_images.append(recon)
            row_images.append(x2)
            all_interpolations.append(torch.cat(row_images, dim=0))
            
    final_grid = torch.cat(all_interpolations, dim=0)
    if not os.path.exists('results3'):
        os.makedirs('results3')
    save_image(final_grid.cpu(), 'results3/interpolation_grid.png', nrow=11)
    print("Interpolation grid saved to results3/interpolation_grid.png")

if __name__ == "__main__":
    interpolate()
