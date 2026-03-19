import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import random

# VAE Model Definition (to match main.py)
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def interpolate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): # For Mac M1/M2/M3
        device = torch.device("mps")
    
    # Load model
    model = VAE().to(device)
    if os.path.exists('vae_mnist.pth'):
        model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
        model.eval()
        print("Model loaded from vae_mnist.pth")
    else:
        print("Error: vae_mnist.pth not found. Please run main.py first to train the model.")
        return

    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=10000, shuffle=False)
    
    data, targets = next(iter(test_loader))
    
    all_interpolations = []
    
    for digit in range(10):
        # Find indices of images belonging to this digit class
        indices = (targets == digit).nonzero(as_tuple=True)[0]
        
        # Pick two random indices
        idx1, idx2 = random.sample(indices.tolist(), 2)
        x1, x2 = data[idx1].unsqueeze(0).to(device), data[idx2].unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu1, _ = model.encode(x1.view(-1, 784))
            mu2, _ = model.encode(x2.view(-1, 784))
            
            row_images = [x1] # Start with x1
            
            # Interpolate lambda from 0.1 to 0.9 (9 steps)
            for i in range(1, 10):
                lam = i / 10.0
                z_lam = (1 - lam) * mu1 + lam * mu2
                recon = model.decode(z_lam).view(1, 1, 28, 28)
                row_images.append(recon)
                
            row_images.append(x2) # End with x2
            
            all_interpolations.append(torch.cat(row_images, dim=0))
            
    # Combine all rows into a single grid (10 rows, 11 columns)
    final_grid = torch.cat(all_interpolations, dim=0)
    
    if not os.path.exists('results'):
        os.makedirs('results')
        
    save_image(final_grid.cpu(), 'results/interpolation_grid.png', nrow=11)
    print("Interpolation grid saved to results/interpolation_grid.png")

if __name__ == "__main__":
    interpolate()
