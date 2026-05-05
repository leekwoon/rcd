"""
RCD rollout entrypoint.

Runs compositional diffusion planning with RCD guidance (self-reconstruction
error as density proxy + overlap consistency), on top of plain arithmetic
overlap averaging for the internal compose operators used by the Tweedie-style
proxy.

Implementation detail:
  1. monkeypatch compose operators on the diffusion class
  2. run the original rollout entrypoint unchanged
"""

import os
import runpy
import sys

sys.path.append("./")
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"

import torch

from diffuser.models.cd_stgl_sml_dfu.stgl_sml_diffusion_v1 import (
    Stgl_Sml_GauDiffusion_InvDyn_V1,
)


def compose_chunk_seq_mean_torch(
    self, chunk_seq: torch.Tensor, beta: float = 2.0
) -> torch.Tensor:
    assert chunk_seq.ndim == 4
    n_comp, batch_size, _, dim = chunk_seq.shape
    total_hzn = self.get_total_hzn(n_comp)
    step = self.horizon - self.len_ovlp_cd

    traj_out = torch.zeros(
        (batch_size, total_hzn, dim),
        device=chunk_seq.device,
        dtype=chunk_seq.dtype,
    )

    for i_c in range(n_comp):
        chunk = chunk_seq[i_c]
        if i_c == 0:
            st_idx = 0
            ed_idx = step
            traj_out[:, st_idx:ed_idx, :] = chunk[:, :step, :]
        elif i_c < n_comp - 1:
            st_idx = self.horizon + (i_c - 1) * step
            ed_idx = st_idx + (self.horizon - 2 * self.len_ovlp_cd)
            traj_out[:, st_idx:ed_idx, :] = chunk[
                :, self.len_ovlp_cd : self.horizon - self.len_ovlp_cd, :
            ]
        else:
            st_idx = self.horizon + (i_c - 1) * step
            ed_idx = st_idx + step
            traj_out[:, st_idx:ed_idx, :] = chunk[:, self.len_ovlp_cd :, :]

    for i_c in range(n_comp - 1):
        st_idx = (i_c + 1) * step
        ed_idx = st_idx + self.len_ovlp_cd
        end_ov = chunk_seq[i_c, :, -self.len_ovlp_cd :, :]
        st_ov = chunk_seq[i_c + 1, :, : self.len_ovlp_cd, :]
        traj_out[:, st_idx:ed_idx, :] = 0.5 * (end_ov + st_ov)

    return traj_out


def compose_window_seq_mean_torch(
    self,
    window_seq: torch.Tensor,
    window_starts,
    total_hzn: int,
    beta: float = 1.0,
) -> torch.Tensor:
    assert window_seq.ndim == 4
    n_windows, batch_size, _, dim = window_seq.shape
    assert n_windows == len(window_starts)

    traj_out = torch.zeros(
        (batch_size, total_hzn, dim),
        device=window_seq.device,
        dtype=window_seq.dtype,
    )
    weight_sum = torch.zeros(
        (batch_size, total_hzn, 1),
        device=window_seq.device,
        dtype=window_seq.dtype,
    )

    for idx_w, st_idx in enumerate(window_starts):
        ed_idx = st_idx + self.horizon
        traj_out[:, st_idx:ed_idx, :] += window_seq[idx_w]
        weight_sum[:, st_idx:ed_idx, :] += 1.0

    return traj_out / torch.clamp(weight_sum, min=1e-6)


def apply_rcd_monkeypatch() -> None:
    Stgl_Sml_GauDiffusion_InvDyn_V1.compose_chunk_seq_exp_torch = (
        compose_chunk_seq_mean_torch
    )
    Stgl_Sml_GauDiffusion_InvDyn_V1.compose_window_seq_torch = (
        compose_window_seq_mean_torch
    )


if __name__ == "__main__":
    apply_rcd_monkeypatch()
    runpy.run_path(
        "./diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py",
        run_name="__main__",
    )
