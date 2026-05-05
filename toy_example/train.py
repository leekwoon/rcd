"""
train.py — Train 3 local diffusion models for the l=3 toy composition experiment.

Models (each takes 3 consecutive positions):
  1) start  : (x_0, x_1, x_2)       — 0 → ±1 → ±1  (same mode for x_1, x_2)
  2) bridge : (x_i, x_{i+1}, x_{i+2}) — ±1 → ±1 → ±1  (all same mode)
  3) end    : (x_{L-3}, x_{L-2}, x_{L-1}) — ±1 → ±1 → 0  (same mode for x_{L-3}, x_{L-2})

Architecture: Simple2DUNet (input_dim=3, hidden=[128,256,512], time_embed=128, blocks=2, dropout=0.1)
DDPM: 1000 steps, beta ∈ [0.0001, 0.02], linear schedule
Training: Adam lr=1e-4, 200 epochs, 5000 samples, batch=256
"""

import argparse
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ──────────────────────────────── Model ────────────────────────────────
# Identical architecture to l=2 version, only input_dim default changed to 3.


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1))
        )
        emb = t[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden)
        self.time_proj = nn.Linear(t_dim, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.linear1(self.norm1(x)) + self.time_proj(t_emb))
        h = self.linear2(self.norm2(self.drop(h)))
        return x + self.drop(h)


class Simple2DUNet(nn.Module):
    """MLP UNet-like architecture for noise prediction."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 256, 512),
        time_embed_dim: int = 128,
        num_blocks_per_level: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_embed_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        for i, hd in enumerate(hidden_dims):
            self.enc_blocks.append(
                nn.ModuleList(
                    [
                        MLPBlock(hd, hd * 2, time_embed_dim, dropout)
                        for _ in range(num_blocks_per_level)
                    ]
                )
            )
            if i < len(hidden_dims) - 1:
                self.enc_downs.append(nn.Linear(hd, hidden_dims[i + 1]))

        # middle
        self.mid = MLPBlock(
            hidden_dims[-1], hidden_dims[-1] * 2, time_embed_dim, dropout
        )

        # decoder
        rev = list(reversed(hidden_dims))
        self.dec_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i, hd in enumerate(rev[:-1]):
            nd = rev[i + 1]
            self.dec_ups.append(nn.Linear(hd, nd))
            self.dec_blocks.append(
                nn.ModuleList(
                    [nn.Linear(nd * 2, nd)]
                    + [
                        MLPBlock(nd, nd * 2, time_embed_dim, dropout)
                        for _ in range(num_blocks_per_level)
                    ]
                )
            )

        self.output_proj = nn.Linear(hidden_dims[0], input_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_emb(timestep)
        x = self.input_proj(x)
        skips = []
        for blks, down in zip(self.enc_blocks[:-1], self.enc_downs):
            for b in blks:
                x = b(x, t)
            skips.append(x)
            x = F.silu(down(x))
        for b in self.enc_blocks[-1]:
            x = b(x, t)
        x = self.mid(x, t)
        for up, blks, skip in zip(self.dec_ups, self.dec_blocks, reversed(skips)):
            x = F.silu(up(x))
            x = blks[0](torch.cat([x, skip], -1))
            for b in blks[1:]:
                x = b(x, t)
        return self.output_proj(x)


# ──────────────────────────────── Dataset ────────────────────────────────


BOUNDARY_STD = 0.2
INTERIOR_STD = 0.1


class TripletDataset(Dataset):
    """Dataset of (x_a, x_b, x_c) triplets sampled from valid mode combinations.

    Args:
        num_samples: Total number of triplets to generate.
        mode_specs: List of valid mode combinations.
                    Each element is a list of 3 (mean, std) tuples.
                    Modes are sampled with equal probability.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int,
        mode_specs: list,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        n_modes = len(mode_specs)
        data = np.empty((num_samples, 3), dtype=np.float32)

        mode_indices = rng.integers(n_modes, size=num_samples)
        for i in range(num_samples):
            spec = mode_specs[mode_indices[i]]
            for j in range(3):
                mean, std = spec[j]
                data[i, j] = rng.normal(mean, std)

        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ──────────────────────────────── Training ────────────────────────────────
# Identical to l=2 version.


def train_one_model(
    model: nn.Module,
    dataset: Dataset,
    scheduler: DDPMScheduler,
    device: torch.device,
    num_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-4,
    save_path: str = "model.pth",
):
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            bs = batch.shape[0]
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bs,), device=device
            )
            noise = torch.randn_like(batch)
            noisy = scheduler.add_noise(batch, noise, timesteps)
            pred = model(noisy, timesteps)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * bs

        avg = epoch_loss / len(dataset)
        if epoch % 20 == 0 or epoch == num_epochs:
            print(f"  epoch {epoch:>3}/{num_epochs}  loss={avg:.6f}")

        if epoch % 50 == 0 or epoch == num_epochs:
            torch.save(model.state_dict(), save_path)

    print(f"  Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train 3 local diffusion models (l=3 triplets)"
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        clip_sample_range=1.5,
    )

    B = BOUNDARY_STD
    I = INTERIOR_STD

    # ── Model 1: start chunk  (0 → ±1 → ±1) ──
    # x_0 ~ N(0, B),  x_1, x_2 ~ same mode from {N(-1, I), N(+1, I)}
    ds_start = TripletDataset(
        num_samples=args.num_samples,
        mode_specs=[
            [(0.0, B), (-1.0, I), (-1.0, I)],
            [(0.0, B), (1.0, I), (1.0, I)],
        ],
        seed=args.seed,
    )

    # ── Model 2: bridge chunk  (±1 → ±1 → ±1, mode-preserving) ──
    ds_bridge = TripletDataset(
        num_samples=args.num_samples,
        mode_specs=[
            [(-1.0, I), (-1.0, I), (-1.0, I)],
            [(1.0, I), (1.0, I), (1.0, I)],
        ],
        seed=args.seed + 1,
    )

    # ── Model 3: end chunk  (±1 → ±1 → 0) ──
    # x_0, x_1 ~ same mode from {N(-1, I), N(+1, I)},  x_2 ~ N(0, B)
    ds_end = TripletDataset(
        num_samples=args.num_samples,
        mode_specs=[
            [(-1.0, I), (-1.0, I), (0.0, B)],
            [(1.0, I), (1.0, I), (0.0, B)],
        ],
        seed=args.seed + 2,
    )

    configs = [
        ("model_start", ds_start),
        ("model_bridge", ds_bridge),
        ("model_end", ds_end),
    ]

    for name, ds in configs:
        print(f"\n{'=' * 60}")
        print(f"Training: {name}  ({len(ds)} samples, input_dim=3)")
        print(f"{'=' * 60}")
        model = Simple2DUNet(input_dim=3)
        save_path = os.path.join(args.save_dir, f"{name}.pth")
        train_one_model(
            model=model,
            dataset=ds,
            scheduler=scheduler,
            device=device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
