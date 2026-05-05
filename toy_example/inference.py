"""
inference.py — long-horizon inference for the l=3 toy composition setting.

Methods:
  - base:    CompDiffuser score averaging
  - rcd:     Reconstruction-coupled density guidance with overlap term
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from tqdm import tqdm

from train import Simple2DUNet


BOUNDARY_STD = 0.2
INTERIOR_STD = 0.1


# ──────────────────────────── Domain assembly ────────────────────────────


class ToyDomain(nn.Module):
    """Loads 3 pretrained local models (l=3) and tiles them across a horizon."""

    def __init__(self, checkpoint_dir: str, device: torch.device, horizon: int = 4):
        super().__init__()
        if horizon < 4:
            raise ValueError(f"Horizon must be >= 4 for l=3 chunks, got {horizon}")

        self.device = device
        num_chunks = horizon - 2
        num_bridges = max(0, num_chunks - 2)

        start = Simple2DUNet(input_dim=3)
        start.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "model_start.pth"),
                map_location=device,
                weights_only=True,
            )
        )

        bridge = Simple2DUNet(input_dim=3)
        bridge.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "model_bridge.pth"),
                map_location=device,
                weights_only=True,
            )
        )

        end = Simple2DUNet(input_dim=3)
        end.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "model_end.pth"),
                map_location=device,
                weights_only=True,
            )
        )

        self.models: List[nn.Module] = [start] + [bridge] * num_bridges + [end]
        for model in self.models:
            model.to(device).eval()

        self.views: List[Tuple[int, int]] = [(i, i + 3) for i in range(num_chunks)]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,
            clip_sample_range=1.5,
        )

    @property
    def total_dim(self) -> int:
        return self.views[-1][1]


# ──────────────────────────── Core helpers ────────────────────────────


def _predict_x0(
    xt: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    return (xt - torch.sqrt(1 - alpha_bar) * noise) / torch.sqrt(alpha_bar)


@torch.no_grad()
def _compdiffuser_noise(
    domain: ToyDomain,
    latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    batch_size = latents.shape[0]
    value = torch.zeros_like(latents)
    count = torch.zeros_like(latents)
    t_batch = timestep.expand(batch_size).to(latents.device)

    for idx, (start, end) in enumerate(domain.views):
        value[:, start:end] += domain.models[idx](latents[:, start:end], t_batch)
        count[:, start:end] += 1

    return torch.where(count > 0, value / count, value)


def _compose_global_x0_from_windows(
    domain: ToyDomain,
    x0_windows: torch.Tensor,
) -> torch.Tensor:
    batch_size = x0_windows.shape[1]
    global_x0 = torch.zeros(
        batch_size,
        domain.total_dim,
        device=x0_windows.device,
        dtype=x0_windows.dtype,
    )
    count = torch.zeros_like(global_x0)
    for idx, (start, end) in enumerate(domain.views):
        global_x0[:, start:end] += x0_windows[idx]
        count[:, start:end] += 1
    return torch.where(count > 0, global_x0 / count, global_x0)


def _compute_overlap_energy_from_x0(x0_preds: torch.Tensor) -> torch.Tensor:
    batch_size = x0_preds.shape[1]
    energy = torch.zeros(batch_size, device=x0_preds.device, dtype=x0_preds.dtype)

    for k in range(x0_preds.shape[0] - 1):
        diff1 = x0_preds[k][:, 1] - x0_preds[k + 1][:, 0]
        diff2 = x0_preds[k][:, 2] - x0_preds[k + 1][:, 1]
        energy = energy + 0.5 * (diff1.pow(2) + diff2.pow(2))

    if x0_preds.shape[0] > 1:
        energy = energy / float(x0_preds.shape[0] - 1)
    return energy


def _probe_timestep(
    scheduler: DDPMScheduler,
    probe_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    probe_t = int(round((scheduler.config.num_train_timesteps - 1) * probe_ratio))
    probe_t = max(1, min(scheduler.config.num_train_timesteps - 1, probe_t))
    return torch.tensor([probe_t], device=device, dtype=torch.long)


def _predict_global_reconstruction(
    domain: ToyDomain,
    global_x0: torch.Tensor,
    probe_t: torch.Tensor,
    clamp_x0: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = global_x0.shape[0]
    device = global_x0.device
    dtype = global_x0.dtype
    alpha_bar_p = domain.scheduler.alphas_cumprod[int(probe_t.item())].to(
        device=device, dtype=dtype
    )
    t_batch = probe_t.expand(batch_size).to(device)
    noise = torch.randn_like(global_x0)
    global_xt = torch.sqrt(alpha_bar_p) * global_x0 + torch.sqrt(1 - alpha_bar_p) * noise

    x0_recons = []
    for idx, (start, end) in enumerate(domain.views):
        local_xt = global_xt[:, start:end]
        eps = domain.models[idx](local_xt, t_batch)
        x0_hat = _predict_x0(local_xt, eps, alpha_bar_p).clamp(-clamp_x0, clamp_x0)
        x0_recons.append(x0_hat)

    x0_recons = torch.stack(x0_recons, dim=0)
    global_recon = _compose_global_x0_from_windows(domain, x0_recons)
    overlap = _compute_overlap_energy_from_x0(x0_recons)
    return global_recon, overlap


def compute_rcd_guidance(
    domain: ToyDomain,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    clamp_x0: float = 1.5,
    probe_ratio: float = 0.35,
    n_mc_samples: int = 1,
    overlap_weight: float = 0.25,
    recon_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = latents.shape[0]
    device = latents.device
    probe_tensor = _probe_timestep(domain.scheduler, probe_ratio, device)

    with torch.enable_grad():
        lat = latents.detach().requires_grad_(True)
        alpha_bar_t = domain.scheduler.alphas_cumprod[int(timestep.item())].to(
            device=device, dtype=lat.dtype
        )
        t_batch = timestep.expand(batch_size).to(device)

        x0_preds = []
        for idx, (start, end) in enumerate(domain.views):
            local_xt = lat[:, start:end]
            eps = domain.models[idx](local_xt, t_batch)
            x0 = _predict_x0(local_xt, eps, alpha_bar_t).clamp(-clamp_x0, clamp_x0)
            x0_preds.append(x0)

        global_x0 = _compose_global_x0_from_windows(domain, torch.stack(x0_preds, dim=0))

        recons = []
        overlaps = []
        for _ in range(max(1, n_mc_samples)):
            global_recon, overlap = _predict_global_reconstruction(
                domain,
                global_x0,
                probe_tensor,
                clamp_x0=clamp_x0,
            )
            recons.append(global_recon)
            overlaps.append(overlap)

        recon_stack = torch.stack(recons, dim=0)
        recon_err = ((recon_stack - global_x0.unsqueeze(0)) ** 2).mean(dim=(0, 2))
        overlap_term = torch.stack(overlaps, dim=0).mean(dim=0)
        energy = recon_weight * recon_err + overlap_weight * overlap_term
        grad = torch.autograd.grad(energy.sum(), lat)[0]

    return grad.detach(), energy.detach()


# ──────────────────────────── Baselines ────────────────────────────


@torch.no_grad()
def sample_compdiffuser(
    domain: ToyDomain,
    batch_size: int = 200,
    num_inference_steps: int = 100,
) -> torch.Tensor:
    device = domain.device
    latents = torch.randn(batch_size, domain.total_dim, device=device)
    domain.scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(domain.scheduler.timesteps, desc="CompDiffuser"):
        latents[:, 0] = 0.0
        latents[:, -1] = 0.0
        combined = _compdiffuser_noise(domain, latents, t)
        latents = domain.scheduler.step(combined, t, latents)["prev_sample"]

    latents[:, 0] = 0.0
    latents[:, -1] = 0.0
    return latents


def _normalize_guidance_grad(grad: torch.Tensor) -> torch.Tensor:
    max_abs = grad.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    return grad / max_abs


def sample_rcd(
    domain: ToyDomain,
    batch_size: int = 200,
    num_inference_steps: int = 100,
    guidance_scale: float = 2.0,
    clamp_x0: float = 1.5,
    probe_ratio: float = 0.35,
    n_mc_samples: int = 1,
    inter_rate: int = 1,
    t_mid_ratio: float = 0.0,
    use_normed_grad: bool = True,
    overlap_weight: float = 0.25,
    recon_weight: float = 1.0,
) -> torch.Tensor:
    device = domain.device
    latents = torch.randn(batch_size, domain.total_dim, device=device)
    domain.scheduler.set_timesteps(num_inference_steps)
    inter_rate = max(1, int(inter_rate))
    cutoff = int(len(domain.scheduler.timesteps) * float(t_mid_ratio))

    for i, t in enumerate(tqdm(domain.scheduler.timesteps, desc="RCD")):
        latents[:, 0] = 0.0
        latents[:, -1] = 0.0

        if i >= cutoff and (i % inter_rate == 0):
            base_noise = _compdiffuser_noise(domain, latents, t)
            grad, _ = compute_rcd_guidance(
                domain,
                latents,
                t,
                clamp_x0=clamp_x0,
                probe_ratio=probe_ratio,
                n_mc_samples=n_mc_samples,
                overlap_weight=overlap_weight,
                recon_weight=recon_weight,
            )
            if use_normed_grad:
                grad = _normalize_guidance_grad(grad)

            step_out = domain.scheduler.step(base_noise, t, latents)
            latents = step_out.prev_sample
            if int(t.item()) > 0:
                variance = domain.scheduler._get_variance(int(t.item())).to(
                    device=latents.device,
                    dtype=latents.dtype,
                )
                latents = latents - guidance_scale * variance * grad
        else:
            combined = _compdiffuser_noise(domain, latents, t)
            latents = domain.scheduler.step(combined, t, latents)["prev_sample"]

    latents[:, 0] = 0.0
    latents[:, -1] = 0.0
    return latents


# ──────────────────────────── Metrics ────────────────────────────


def evaluate_valid(samples: np.ndarray, num_stds: float = 3.0) -> np.ndarray:
    boundary_r = num_stds * BOUNDARY_STD
    interior_r = num_stds * INTERIOR_STD
    start_ok = np.abs(samples[:, 0]) < boundary_r
    end_ok = np.abs(samples[:, -1]) < boundary_r
    interior = samples[:, 1:-1]
    interior_ok = np.all(
        np.minimum(np.abs(interior - 1.0), np.abs(interior + 1.0)) < interior_r,
        axis=1,
    )
    signs = np.sign(interior)
    signs[signs == 0] = 1
    same_mode = np.all(signs == signs[:, [0]], axis=1)
    return start_ok & end_ok & interior_ok & same_mode


def summarize_samples(samples: np.ndarray) -> Dict[str, float]:
    valid = evaluate_valid(samples)
    interior = samples[:, 1:-1]
    signs = np.sign(interior)
    signs[signs == 0] = 1
    switch = np.any(signs != signs[:, [0]], axis=1)
    mode_mean = interior.mean(axis=1) if interior.size else np.zeros(samples.shape[0])
    overall_pos = int((mode_mean > 0).sum())
    overall_neg = int((mode_mean <= 0).sum())
    valid_pos = int(((mode_mean > 0) & valid).sum())
    valid_neg = int(((mode_mean <= 0) & valid).sum())
    valid_total = max(valid_pos + valid_neg, 1)
    valid_balance_ratio = float(min(valid_pos, valid_neg) / max(valid_pos, valid_neg, 1))
    summary = {
        "success_rate": float(valid.mean()),
        "mode_switch_rate": float(switch.mean()),
        "endpoint_abs_mean": float(np.abs(samples[:, [0, -1]]).mean()),
        "interior_abs_mean": float(np.abs(interior).mean()) if interior.size else 0.0,
        "overall_positive_count": overall_pos,
        "overall_negative_count": overall_neg,
        "valid_positive_count": valid_pos,
        "valid_negative_count": valid_neg,
        "valid_mode_balance_ratio": valid_balance_ratio,
        "valid_mode_imbalance": float(abs(valid_pos - valid_neg) / valid_total),
        "mode_collapse_valid": bool((valid_pos == 0) or (valid_neg == 0)),
    }
    return summary


# ──────────────────────────── CLI ────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Toy l=3 inference")
    parser.add_argument(
        "--method",
        type=str,
        default="base",
        choices=["base", "rcd"],
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--rcd-guidance-scale", type=float, default=2.0)
    parser.add_argument("--rcd-clamp-x0", type=float, default=1.5)
    parser.add_argument("--rcd-probe-ratio", type=float, default=0.35)
    parser.add_argument("--rcd-n-mc-samples", type=int, default=1)
    parser.add_argument("--rcd-inter-rate", type=int, default=1)
    parser.add_argument("--rcd-t-mid-ratio", type=float, default=0.0)
    parser.add_argument("--rcd-use-normed-grad", type=int, default=1)
    parser.add_argument("--rcd-overlap-weight", type=float, default=1.0)
    parser.add_argument("--rcd-recon-weight", type=float, default=1.0)
    parser.add_argument("--method-tag", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, method={args.method}, L={args.horizon}")

    domain = ToyDomain(args.checkpoint_dir, device, horizon=args.horizon)

    if args.method == "base":
        samples = sample_compdiffuser(domain, args.batch_size, args.steps)
    elif args.method == "rcd":
        samples = sample_rcd(
            domain,
            args.batch_size,
            args.steps,
            guidance_scale=args.rcd_guidance_scale,
            clamp_x0=args.rcd_clamp_x0,
            probe_ratio=args.rcd_probe_ratio,
            n_mc_samples=args.rcd_n_mc_samples,
            inter_rate=args.rcd_inter_rate,
            t_mid_ratio=args.rcd_t_mid_ratio,
            use_normed_grad=bool(args.rcd_use_normed_grad),
            overlap_weight=args.rcd_overlap_weight,
            recon_weight=args.rcd_recon_weight,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    samples_np = samples.float().cpu().numpy()
    summary = summarize_samples(samples_np)

    os.makedirs(args.save_dir, exist_ok=True)
    suffix = f"_{args.method_tag}" if args.method_tag else ""
    tag = f"L{args.horizon}_{args.method}{suffix}"
    out_npy = os.path.join(args.save_dir, f"{tag}.npy")
    out_json = os.path.join(args.save_dir, f"{tag}.json")
    np.save(out_npy, samples_np)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {samples_np.shape} -> {out_npy}")
    print(f"Summary -> {out_json}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
