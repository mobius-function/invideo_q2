# models/diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Add time embedding
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class EmbeddingConditionedUNet(nn.Module):
    """
    A U-Net model that takes embeddings as a conditioning input.
    """

    def __init__(self, embedding_dim=512, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device

        # Embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(embedding_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList([
            Block(64, 128, time_dim),
            Block(128, 256, time_dim),
            Block(256, 512, time_dim),
            Block(512, 512, time_dim),
        ])

        # Bottleneck
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder
        self.ups = nn.ModuleList([
            Block(512, 512, time_dim, up=True),
            Block(512, 256, time_dim, up=True),
            Block(256, 128, time_dim, up=True),
            Block(128, 64, time_dim, up=True),
        ])

        # Final output
        self.output = nn.Conv2d(64, 3, 1)

    def forward(self, x, timestep, embedding):
        # Time embedding
        t = self.time_mlp(timestep)

        # Embedding projection
        emb = self.embedding_proj(embedding)

        # Combine embeddings
        combined_embedding = t + emb

        # Initial convolution
        x = self.conv0(x)

        # Cache residuals for skip connections
        residuals = []

        # Encoder
        for down in self.downs:
            residuals.append(x)
            x = down(x, combined_embedding)

        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Decoder with skip connections
        for up, residual in zip(self.ups, residuals[::-1]):
            x = torch.cat((x, residual), dim=1)
            x = up(x, combined_embedding)

        return self.output(x)


class DiffusionModel:
    def __init__(
            self,
            model,
            noise_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            img_size=128,
            device="cuda"
    ):
        self.model = model
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Define beta schedule
        self.betas = self.prepare_noise_schedule().to(device)
        # Define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # Pre-calculate diffusion parameters
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        # Get noise
        noise = torch.randn_like(x)

        # Add noise to image
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def sample(self, embedding, n=1, cfg_scale=3.0):
        self.model.eval()

        # Start from pure noise
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

        # Prepare unconditional embedding for classifier-free guidance
        if cfg_scale > 1.0:
            uncond_embedding = torch.zeros_like(embedding).to(self.device)

        # Sampling loop
        for i in reversed(range(self.noise_steps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)

            # Predict noise
            if cfg_scale > 1.0:
                # Classifier-free guidance with conditional and unconditional predictions
                predicted_noise_cond = self.model(x, t, embedding)
                predicted_noise_uncond = self.model(x, t, uncond_embedding)
                predicted_noise = torch.lerp(predicted_noise_uncond, predicted_noise_cond, cfg_scale)
            else:
                predicted_noise = self.model(x, t, embedding)

            # Get alpha values for current timestep
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            # Only add noise if we're not at the last step
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Sample from posterior distribution
            x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        # Ensure output is in proper range
        x = torch.clamp(x, -1, 1)
        self.model.train()
        return x

    def train_step(self, optimizer, images, embeddings):
        # Get batch size
        batch_size = images.shape[0]

        # Reset gradients
        optimizer.zero_grad()

        # Sample random timesteps
        t = torch.randint(0, self.noise_steps, (batch_size,), device=self.device).long()

        # Add noise to images
        x_noisy, noise = self.noise_images(images, t)

        # Predict noise
        predicted_noise = self.model(x_noisy, t, embeddings)

        # Calculate loss (mean squared error)
        loss = F.mse_loss(noise, predicted_noise)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        return loss.item()