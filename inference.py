# inference.py
import torch
import argparse
import os
from PIL import Image
from torchvision.utils import save_image, make_grid
from models.embedding import FaceEmbedder
from models.diffusion import EmbeddingConditionedUNet, DiffusionModel
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate face images using trained diffusion model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to reference image for embedding extraction')
    parser.add_argument('--output_dir', type=str, default='./generated', help='Output directory')
    parser.add_argument('--samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--cfg_scale', type=float, default=3.0, help='Classifier-free guidance scale')
    parser.add_argument('--noise_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize embedding model
    embedder = FaceEmbedder(device=args.device)

    # Initialize U-Net model
    model = EmbeddingConditionedUNet(
        embedding_dim=512,
        time_dim=256,
        device=device
    ).to(device)

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize diffusion
    diffusion = DiffusionModel(
        model=model,
        noise_steps=args.noise_steps,
        img_size=args.img_size,
        device=device
    )

    # Extract embedding from reference image or use random embedding
    if args.image_path:
        reference_img = Image.open(args.image_path).convert('RGB')
        embedding, _ = embedder.get_embedding(reference_img)

        if embedding is None:
            print("No face detected in reference image. Using random embedding.")
            embedding = torch.randn(1, 512).to(device)
        else:
            # Save reference image
            reference_img = reference_img.resize((args.img_size, args.img_size))
            reference_img.save(os.path.join(args.output_dir, 'reference.png'))
    else:
        print("No reference image provided. Using random embedding.")
        embedding = torch.randn(1, 512).to(device)

    # Repeat embedding for multiple samples
    embedding = embedding.repeat(args.samples, 1)

    # Generate samples
    print(f"Generating {args.samples} samples...")
    samples = diffusion.sample(embedding, n=args.samples, cfg_scale=args.cfg_scale)

    # Denormalize samples
    samples = samples * 0.5 + 0.5

    # Save individual samples
    for i, sample in enumerate(samples):
        save_image(sample, os.path.join(args.output_dir, f'sample_{i + 1}.png'))

    # Create and save grid
    grid = make_grid(samples, nrow=int(np.sqrt(args.samples)))
    save_image(grid, os.path.join(args.output_dir, 'samples_grid.png'))

    print(f"Samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()