import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict

sys.path.append("./")

import numpy as np

from diffuser.ogb_task.ogb_maze_v1.eval_plan_quality import (
    combine_wall_stats,
    get_plan_quality_tracks,
    wall_penetration_stats,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


def ogb_xy_to_ij_noenv(xy_trajs):
    """
    Lightweight OGBench xy->ij conversion without constructing a MuJoCo env.
    OGBench locomaze uses maze_unit=4, offset_x=4, offset_y=4.
    """
    assert xy_trajs.ndim in [2, 3] and xy_trajs.shape[-1] == 2
    maze_unit = 4.0
    offset_x = 4.0
    offset_y = 4.0
    i = (xy_trajs[..., 1:2] + offset_y + 0.5 * maze_unit) / maze_unit
    j = (xy_trajs[..., 0:1] + offset_x + 0.5 * maze_unit) / maze_unit
    return np.concatenate([i, j], axis=-1) - 0.5


def parse_maze_grid_noenv(env_name):
    """
    Maze-spec lookup without importing the full ogbench package.
    """
    maze_large = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    maze_medium = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    maze_giant = "################\\#O#OOOOOO##OOOO#\\#O#O##O#O#OO##O#\\#OOO#OO#OOO#OOO#\\#O###O######O#O#\\#OOO#OOO#OOOO#O#\\###O#O#OO#O#O###\\#OOO#OO#OOO#OOO#\\#O#O#O######O#O#\\#O###OOO#OOO##O#\\#OOOOO#OOO#OOOO#\\################"
    maze_arena = "########\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\########"
    env_name = env_name.lower()
    if "large" in env_name:
        maze_string = maze_large
    elif "medium" in env_name:
        maze_string = maze_medium
    elif "giant" in env_name:
        maze_string = maze_giant
    elif "arena" in env_name:
        maze_string = maze_arena
    else:
        raise ValueError(f"Unknown maze env for grid lookup: {env_name}")
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def main():
    args = parse_args()
    from diffuser.utils.cp_utils.cp_serial import load_stgl_lh_ev_probs_hdf5
    from diffuser.utils.ogb_utils.ogb_serial import get_ogb_maze_ev_probs_fname

    run_dir = os.path.abspath(args.run_dir)
    json_path = os.path.join(run_dir, "00_rollout.json")
    pkl_path = os.path.join(run_dir, "00_rout.pkl")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing rollout json: {json_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing rollout pkl: {pkl_path}")

    rollout = json.load(open(json_path, "r"))
    rollout_pkl = pickle.load(open(pkl_path, "rb"))

    maze_grid = parse_maze_grid_noenv(args.env)
    probs_h5 = get_ogb_maze_ev_probs_fname(args.env)
    probs = load_stgl_lh_ev_probs_hdf5(probs_h5)

    ep_st_idx = int(rollout.get("ep_st_idx", 0))
    ep_is_suc = rollout.get("ep_is_suc", [])
    pred_full = rollout_pkl["ep_pred_obss_full"]
    n_eval = min(len(ep_is_suc), len(pred_full))

    per_episode = []
    wall_ratios = []

    for local_idx in range(n_eval):
        actual_ep = ep_st_idx + local_idx
        pick_traj = np.asarray(pred_full[local_idx])
        st_state = probs["start_state"][actual_ep]
        gl_pos = probs["goal_pos"][actual_ep]

        track_specs = get_plan_quality_tracks(args.env, pick_traj, st_state, gl_pos)
        track_results = []
        for track in track_specs:
            traj_ij = ogb_xy_to_ij_noenv(track["traj_xy"])
            st_ij = ogb_xy_to_ij_noenv(track["start_xy"].reshape(1, 2))[0]
            gl_ij = ogb_xy_to_ij_noenv(track["goal_xy"].reshape(1, 2))[0]
            track_results.append(
                dict(
                    name=track["name"],
                    start_ij=st_ij.tolist(),
                    goal_ij=gl_ij.tolist(),
                    wall_stats=wall_penetration_stats(traj_ij, maze_grid),
                )
            )

        ws = combine_wall_stats([track["wall_stats"] for track in track_results])
        wall_ratios.append(float(ws["wall_ratio"]))
        per_episode.append(
            OrderedDict(
                local_idx=local_idx,
                episode_idx=actual_ep,
                success=bool(ep_is_suc[local_idx]),
                wall_ratio=float(ws["wall_ratio"]),
                n_wall=int(ws["n_wall"]),
                n_points=int(ws["n_points"]),
                clean=bool(ws["wall_ratio"] == 0.0),
                tracks=track_results,
            )
        )

    n_clean = int(sum(ep["clean"] for ep in per_episode))
    n_success = int(sum(ep["success"] for ep in per_episode))

    summary = OrderedDict(
        env=args.env,
        run_dir=run_dir,
        num_ep=n_eval,
        ep_st_idx=ep_st_idx,
        success_rate=float(n_success / max(n_eval, 1)),
        n_success=n_success,
        n_clean_plans=n_clean,
        wall_ratio_mean=float(np.mean(wall_ratios)) if wall_ratios else 0.0,
        wall_ratio_max=float(np.max(wall_ratios)) if wall_ratios else 0.0,
        eval_method=rollout.get("eval_method"),
        ev_meta_method=rollout.get("ev_meta_method"),
        ev_global_density_weight=rollout.get("ev_global_density_weight"),
        ev_global_density_inter_rate=rollout.get("ev_global_density_inter_rate"),
        ev_global_density_n_mc=rollout.get("ev_global_density_n_mc"),
        ev_global_density_proxy_type=rollout.get("ev_global_density_proxy_type"),
        ev_global_density_proxy_overlap_weight=rollout.get(
            "ev_global_density_proxy_overlap_weight"
        ),
        per_episode=per_episode,
    )

    out_path = os.path.join(run_dir, "00_subset_quality.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "[subset_quality]",
        f"clean={n_clean}/{n_eval}",
        f"success={n_success}/{n_eval}",
        f"wall_mean={summary['wall_ratio_mean']:.4f}",
        f"-> {out_path}",
    )


if __name__ == "__main__":
    main()
