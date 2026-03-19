import os
import math
import glob
import argparse
import torch
from torchvision.utils import save_image

from model import MNISTDiffusion
from utils import ExponentialMovingAverage


def parse_args():
    parser = argparse.ArgumentParser(description="Sample all checkpoints with DDIM")
    parser.add_argument("--ckpt_dir", type=str, default="DDPMresults", help="directory containing .pt checkpoints")
    parser.add_argument("--out_dir", type=str, default="DDIMresults", help="directory to save generated images")
    parser.add_argument("--n_samples", type=int, default=36, help="number of images to generate per checkpoint")
    parser.add_argument("--model_base_dim", type=int, default=64, help="base dim of Unet")
    parser.add_argument("--timesteps", type=int, default=1000, help="training diffusion timesteps")
    parser.add_argument("--ddim_steps", type=int, default=20, help="number of DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta; 0.0 = deterministic")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    return parser.parse_args()


def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_step_number(path):
    name = os.path.basename(path)
    # expects names like steps_00004690.pt
    try:
        return int(name.replace("steps_", "").replace(".pt", ""))
    except ValueError:
        return float("inf")


def main(args):
    device = get_device(args.cpu)
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_paths = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "steps_*.pt")),
        key=extract_step_number
    )

    if not ckpt_paths:
        print(f"No checkpoint files found in {args.ckpt_dir}")
        return

    print(f"Found {len(ckpt_paths)} checkpoints")

    for ckpt_path in ckpt_paths:
        step_num = extract_step_number(ckpt_path)
        print(f"\nLoading checkpoint: {ckpt_path}")

        model = MNISTDiffusion(
            timesteps=args.timesteps,
            image_size=28,
            in_channels=1,
            base_dim=args.model_base_dim,
            dim_mults=[2, 4],
        ).to(device)

        model_ema = ExponentialMovingAverage(model, device=device, decay=0.995)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["model_ema"])

        model_ema.eval()

        samples = model_ema.module.ddim_sampling(
            n_samples=args.n_samples,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
            device=device,
        )

        out_path = os.path.join(args.out_dir, f"steps_{step_num:08d}.png")
        save_image(samples, out_path, nrow=int(math.sqrt(args.n_samples)))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)