import sys
import os

sys.path.append("./")
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"

import json
import time
import copy
from collections import OrderedDict

import numpy as np

import diffuser.utils as utils
from diffuser.datasets.ogb_dset.ogb_utils import ogb_xy_to_ij
from diffuser.ogb_task.ogb_maze_v1.eval_plan_quality import (
    apply_eval_method_alias,
    apply_rcd_defaults,
    combine_wall_stats,
    get_plan_quality_tracks,
    parse_maze_grid,
    setup_args,
    wall_penetration_stats,
)
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import (
    OgB_Stgl_Sml_MazeEnvPlanner_V1,
)


def main(args_train, args):
    planner = OgB_Stgl_Sml_MazeEnvPlanner_V1(args_train, args=args)
    planner.setup_load(ld_config={})

    pl_seed = None if args.pl_seeds[0] == -1 else args.pl_seeds[0]
    if pl_seed is not None:
        utils.set_seed(pl_seed)

    env = planner.env
    env_name = planner.args.dataset
    obs_sel = planner.obs_select_dim
    maze_grid = parse_maze_grid(env_name)

    total_probs = len(planner.problems_dict["start_state"])
    ep_st_idx = int(getattr(args, "ep_st_idx", 0))
    num_ep = total_probs if args.plan_n_ep == -100 else int(args.plan_n_ep)
    eval_indices = list(range(ep_st_idx, min(ep_st_idx + num_ep, total_probs)))

    seed_tag = f"sd{pl_seed}" if pl_seed is not None else "nosd"
    sub_dir = (
        f"{time.strftime('%y%m%d-%H%M%S')}-"
        f"{args.eval_method}-subset-"
        f"n{len(eval_indices)}-st{ep_st_idx}"
    )
    out_dir = os.path.join(
        args.pq_logbase,
        env_name,
        f"plan_subset_quality_{args.eval_method}",
        sub_dir,
        f"plan_subset_quality_{seed_tag}_{time.strftime('%y%m%d-%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"\n[eval_plan_subset_quality] method={args.eval_method} env={env_name} "
        f"seed={pl_seed} ep_st_idx={ep_st_idx} total={len(eval_indices)}"
    )
    print(f"[eval_plan_subset_quality] output -> {out_dir}\n")

    all_results = []
    for i_ep in eval_indices:
        st_state = planner.problems_dict["start_state"][i_ep]
        gl_pos = planner.problems_dict["goal_pos"][i_ep]

        input_st = st_state[list(obs_sel)]
        gl_for_dfu = gl_pos[list(obs_sel)]
        g_cond = {
            "st_gl": np.array([input_st[None,], gl_for_dfu[None,]], dtype=np.float32),
        }

        t0 = time.time()
        m_out = planner.policy.gen_cond_stgl(g_cond=g_cond, b_s=planner.b_size_per_prob)
        dt = time.time() - t0
        pick_traj = m_out.pick_traj

        track_specs = get_plan_quality_tracks(env_name, pick_traj, st_state, gl_pos)
        track_results = []
        for track in track_specs:
            traj_ij = ogb_xy_to_ij(env, track["traj_xy"])
            track_results.append(
                dict(
                    name=track["name"],
                    wall_stats=wall_penetration_stats(traj_ij, maze_grid),
                )
            )

        ws = combine_wall_stats([track["wall_stats"] for track in track_results])
        ep_result = dict(
            i_ep=i_ep,
            plan_time_s=round(dt, 3),
            plan_len=len(pick_traj),
            wall_ratio=round(ws["wall_ratio"], 4),
            n_wall=int(ws["n_wall"]),
            n_points=int(ws["n_points"]),
            clean=bool(ws["wall_ratio"] == 0.0),
            wall_tracks=[
                dict(
                    name=track["name"],
                    wall_ratio=round(track["wall_stats"]["wall_ratio"], 4),
                    n_wall=int(track["wall_stats"]["n_wall"]),
                    n_points=int(track["wall_stats"]["n_points"]),
                )
                for track in track_results
            ],
            stitch_info=copy.deepcopy(getattr(planner.policy, "last_stitch_info", {})),
        )
        all_results.append(ep_result)

        print(
            f"  ep {i_ep:3d} wall={ws['wall_ratio']:5.1%} "
            f"t={dt:.2f}s {'OK' if ws['wall_ratio'] == 0.0 else 'WALL'}"
        )

    wall_ratios = [r["wall_ratio"] for r in all_results]
    times = [r["plan_time_s"] for r in all_results]
    summary = OrderedDict(
        method_name=args.eval_method,
        meta_method=args.ev_meta_method,
        env=env_name,
        seed=pl_seed,
        ep_st_idx=ep_st_idx,
        n_problems=len(eval_indices),
        wall_ratio_mean=round(float(np.mean(wall_ratios)), 4),
        wall_ratio_std=round(float(np.std(wall_ratios)), 4),
        wall_ratio_max=round(float(np.max(wall_ratios)), 4),
        n_clean_plans=int(sum(1 for r in wall_ratios if r == 0.0)),
        plan_time_mean_s=round(float(np.mean(times)), 3),
    )
    summary["stitch_info"] = getattr(planner.policy, "last_stitch_info", {})
    summary["ev_global_density_weight"] = getattr(args, "ev_global_density_weight", None)
    summary["ev_global_density_inter_rate"] = getattr(
        args, "ev_global_density_inter_rate", None
    )
    summary["ev_global_density_n_mc"] = getattr(args, "ev_global_density_n_mc", None)
    summary["ev_global_density_proxy_type"] = getattr(
        args, "ev_global_density_proxy_type", None
    )
    summary["ev_global_density_proxy_overlap_weight"] = getattr(
        args, "ev_global_density_proxy_overlap_weight", None
    )

    json_path = os.path.join(out_dir, "plan_subset_quality_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    detail_path = os.path.join(out_dir, "plan_subset_quality_detail.json")
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[summary] {json_path}")
    print(f"[detail]  {detail_path}")
    print(
        f"[clean] {summary['n_clean_plans']}/{summary['n_problems']}  "
        f"wall_mean={summary['wall_ratio_mean']:.4f}"
    )


if __name__ == "__main__":
    args_train, args, _ = setup_args()
    args.pq_logbase = getattr(args, "pq_logbase", "logs_pq_subset")

    loadpath = args.logbase, args.dataset, args_train.exp_name
    if getattr(args, "save_logbase", None):
        rel_savepath = os.path.relpath(args.savepath, start=args.logbase)
        args.savepath = os.path.join(args.save_logbase, rel_savepath)
        args.savepath_root = args.save_logbase

    main(args_train, args)
