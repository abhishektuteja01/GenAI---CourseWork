import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import MNISTDiffusion
from utils import ExponentialMovingAverage


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
    ])

    train_dataset = MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess,
    )
    test_dataset = MNIST(
        root="./mnist_data",
        train=False,
        download=True,
        transform=preprocess,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ckpt", type=str, help="define checkpoint path", default="")
    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument("--model_base_dim", type=int, help="base dim of Unet", default=64)
    parser.add_argument("--timesteps", type=int, help="sampling steps of DDPM", default=1000)
    parser.add_argument(
        "--model_ema_steps",
        type=int,
        help="ema model evaluation interval",
        default=10,
    )
    parser.add_argument(
        "--model_ema_decay",
        type=float,
        help="ema model decay",
        default=0.995,
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequency",
        default=10,
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="set to normal sampling method without clip x_0 which could yield unstable samples",
    )
    parser.add_argument("--cpu", action="store_true", help="cpu training")
    return parser.parse_args()


def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for image, _ in dataloader:
            image = image.to(device)
            noise = torch.randn_like(image).to(device)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def plot_losses(train_losses, test_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST Diffusion: Train vs Test Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    device = get_device(args.cpu)
    print(f"Using device: {device}")

    os.makedirs("DDPMresults", exist_ok=True)

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size,
        image_size=28,
    )

    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
    ).to(device)

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * len(train_dataloader),
        pct_start=0.25,
        anneal_strategy="cos",
    )
    loss_fn = nn.MSELoss(reduction="mean")

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    train_losses = []
    test_losses = []
    global_steps = 0

    for i in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for j, (image, _) in enumerate(train_dataloader):
            image = image.to(device)
            noise = torch.randn_like(image).to(device)

            pred = model(image, noise)
            loss = loss_fn(pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)

            global_steps += 1
            epoch_train_loss += loss.item()

            if j % args.log_freq == 0:
                print(
                    "Epoch[{}/{}], Step[{}/{}], Loss:{:.5f}, LR:{:.5f}".format(
                        i + 1,
                        args.epochs,
                        j,
                        len(train_dataloader),
                        loss.item(),
                        scheduler.get_last_lr()[0],
                    )
                )

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        avg_test_loss = evaluate(model, test_dataloader, loss_fn, device)
        test_losses.append(avg_test_loss)

        print(
            "====> Epoch: {} Train Loss: {:.5f} | Test Loss: {:.5f}".format(
                i + 1,
                avg_train_loss,
                avg_test_loss,
            )
        )

        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
        }
        torch.save(ckpt, "DDPMresults/steps_{:0>8}.pt".format(global_steps))

        model_ema.eval()
        samples = model_ema.module.sampling(
            args.n_samples,
            clipped_reverse_diffusion=not args.no_clip,
            device=device,
        )
        save_image(
            samples,
            "DDPMresults/steps_{:0>8}.png".format(global_steps),
            nrow=int(math.sqrt(args.n_samples)),
        )

    plot_losses(train_losses, test_losses, "DDPMresults/loss_plot.png")
    print("Saved loss plot to DDPMresults/loss_plot.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
