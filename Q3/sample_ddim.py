import os
import math
import argparse
import torch
from torchvision.utils import save_image

from model import MNISTDiffusion
from utils import ExponentialMovingAverage


def parse_args():
    parser = argparse.ArgumentParser(description="DDIM Sampling on MNIST")
    parser.add_argument("--ckpt", type=str, required=True, help="path to trained checkpoint")
    parser.add_argument("--n_samples", type=int, default=36)
    parser.add_argument("--model_base_dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(args):
    device = get_device(args.cpu)
    print(f"Using device: {device}")

    os.makedirs("DDIMresults", exist_ok=True)

    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
    ).to(device)

    model_ema = ExponentialMovingAverage(model, device=device, decay=0.995)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model_ema.load_state_dict(ckpt["model_ema"])

    model_ema.eval()

    samples = model_ema.module.ddim_sampling(
        n_samples=args.n_samples,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        device=device,
    )

    save_path = f"DDIMresults/ddim_steps_{args.ddim_steps}.png"
    save_image(samples, save_path, nrow=int(math.sqrt(args.n_samples)))
    print(f"Saved DDIM samples to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)