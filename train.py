import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.cuda.amp import GradScaler, autocast  # Added for mixed precision training
from models.diffusion import EmbeddingConditionedUNet, DiffusionModel
from utils.data_loader import FaceDataset, create_dataloaders
import wandb
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model for face generation')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with face images')
    parser.add_argument('--embedding_dir', type=str, help='Directory for embeddings')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--noise_steps', type=int, default=500, help='Number of noise steps')  # Reduced from 1000
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--sample_interval', type=int, default=10, help='Interval between image samples')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Use mixed precision training')
    return parser.parse_args()


# Modified DiffusionModel train_step method with mixed precision support
def train_step(diffusion, optimizer, images, embeddings, scaler=None):
    # Get batch size
    batch_size = images.shape[0]

    # Reset gradients
    optimizer.zero_grad()

    # Sample random timesteps
    t = torch.randint(0, diffusion.noise_steps, (batch_size,), device=diffusion.device).long()

    # Add noise to images
    x_noisy, noise = diffusion.noise_images(images, t)

    if scaler is not None:
        # Mixed precision training path
        with autocast():
            # Predict noise
            predicted_noise = diffusion.model(x_noisy, t, embeddings)

            # Calculate loss (mean squared error)
            loss = nn.functional.mse_loss(noise, predicted_noise)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard full-precision path
        # Predict noise
        predicted_noise = diffusion.model(x_noisy, t, embeddings)

        # Calculate loss (mean squared error)
        loss = nn.functional.mse_loss(noise, predicted_noise)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

    return loss.item()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

    # Initialize wandb
    if args.wandb:
        wandb.init(project="face-diffusion", name=f"diffusion-{args.img_size}")
        wandb.config.update(args)

    # Set up data loaders
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        args.embedding_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=4,  # Increased from default
        pin_memory=True  # Enable pin_memory for faster data transfer to GPU
    )

    # Initialize model
    model = EmbeddingConditionedUNet(
        embedding_dim=512,  # Face embedding dimension
        time_dim=256,  # Time embedding dimension
        device=device
    ).to(device)

    # Initialize diffusion
    diffusion = DiffusionModel(
        model=model,
        noise_steps=args.noise_steps,
        img_size=args.img_size,
        device=device
    )

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training")

    # Get a fixed batch for sampling
    val_embeddings, _ = next(iter(val_loader))
    val_embeddings = val_embeddings[:8].to(device)  # Use 8 fixed samples

    # Train loop
    start_time = time.time()
    total_time_limit = 6 * 60 * 60  # 6 hours in seconds

    for epoch in range(args.epochs):
        # Check if we've exceeded time limit
        if time.time() - start_time > total_time_limit:
            print(f"Reached 6-hour time limit after {epoch} epochs")
            break

        model.train()
        epoch_loss = 0

        # Progress bar
        pbar = tqdm(train_loader)
        for i, (embeddings, images) in enumerate(pbar):
            embeddings = embeddings.to(device)
            images = images.to(device)

            # Train step with mixed precision support
            loss = train_step(diffusion, optimizer, images, embeddings, scaler)

            epoch_loss += loss

            # Update progress bar
            pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.4f}")

            # Log to wandb
            if args.wandb and i % 10 == 0:
                wandb.log({"loss": loss})

        # Update learning rate
        scheduler.step()

        # Log epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

        if args.wandb:
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})

        # Sample images periodically
        if (epoch + 1) % args.sample_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Sample from model
                samples = diffusion.sample(val_embeddings, cfg_scale=3.0)

                # Denormalize
                samples = samples * 0.5 + 0.5

                # Save samples
                sample_grid = make_grid(samples, nrow=4)
                save_image(sample_grid, os.path.join(args.output_dir, 'samples', f"sample_epoch_{epoch + 1}.png"))

                if args.wandb:
                    wandb.log({"samples": wandb.Image(sample_grid)})

        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pt'))

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.output_dir, 'final_model.pt'))

    # Print total training time
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    main()

