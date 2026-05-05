"""
Planning-Time Benchmark for Rollout Evaluation
==============================================
Measure the wall-clock time of a single planning call,
`planner.policy.gen_cond_stgl(...)`, under the same setup used by rollout eval.

The benchmark:
  - loads the checkpoint once,
  - selects one evaluation problem,
  - runs warmup iterations,
  - repeats the same planning query N times,
  - reports mean/std for full planning time.

It also records the policy's internal sampling-only timing when available.
"""

import sys, os

sys.path.append("./")
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"
import copy
import json
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch

import diffuser.utils as utils
from diffuser.ogb_task.ogb_maze_v1.eval_plan_quality import setup_args


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args_train, args, _ = setup_args()

    pl_seed = args.pl_seeds[0]
    if pl_seed == -1:
        pl_seed = None

    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import (
        OgB_Stgl_Sml_MazeEnvPlanner_V1,
    )

    seed_tag = f"sd{pl_seed}" if pl_seed is not None else "nosd"
    time_dir = (
        f"{datetime.now().strftime('%y%m%d-%H%M%S-%f')[:-3]}"
        f"-{args.eval_method}"
        f"-time-ncp{args.ev_n_comp}"
        f"-ep{args.bench_problem_idx}"
        f"-rep{args.bench_n_repeat}"
        f"-{seed_tag}"
    )
    args.savepath = os.path.join(
        args.time_logbase,
        args.dataset,
        f"plan_time_{args.eval_method}",
        time_dir,
    )

    planner = OgB_Stgl_Sml_MazeEnvPlanner_V1(
        copy.deepcopy(args_train), args=copy.deepcopy(args)
    )
    planner.setup_load(ld_config={})

    total_probs = len(planner.problems_dict["start_state"])
    i_ep = int(np.clip(args.bench_problem_idx, 0, total_probs - 1))

    st_state = planner.problems_dict["start_state"][i_ep]
    gl_pos = planner.problems_dict["goal_pos"][i_ep]
    obs_sel = planner.obs_select_dim

    input_st = st_state[list(obs_sel)]
    gl_for_dfu = gl_pos[list(obs_sel)]
    g_cond = {
        "st_gl": np.array([input_st[None,], gl_for_dfu[None,]], dtype=np.float32),
    }

    print(
        f"\n[eval_plan_time] method={args.eval_method} env={args.dataset} seed={pl_seed} "
        f"problem_idx={i_ep} warmup={args.bench_warmup} repeat={args.bench_n_repeat}"
    )
    print(f"[eval_plan_time] output → {planner.savepath}\n")

    warmup_records = []
    for i_wm in range(args.bench_warmup):
        rep_seed = None
        if pl_seed is not None:
            rep_seed = int(pl_seed + i_wm * args.bench_seed_stride)
            utils.set_seed(rep_seed)

        prev_n = len(planner.policy.ncp_pred_time_list)
        sync_cuda()
        t0 = time.perf_counter()
        m_out = planner.policy.gen_cond_stgl(g_cond=g_cond, b_s=planner.b_size_per_prob)
        sync_cuda()
        dt = time.perf_counter() - t0

        sample_dt = None
        if len(planner.policy.ncp_pred_time_list) > prev_n:
            sample_dt = float(planner.policy.ncp_pred_time_list[-1][1])

        warmup_records.append(
            dict(
                warmup_idx=i_wm,
                seed=rep_seed,
                plan_time_s=float(dt),
                sample_time_s=sample_dt,
                plan_len=int(len(m_out.pick_traj)),
            )
        )
        print(f"  warmup {i_wm:2d}  t={dt:.4f}s")

    records = []
    for i_rep in range(args.bench_n_repeat):
        rep_seed = None
        if pl_seed is not None:
            rep_seed = int(
                pl_seed + (args.bench_warmup + i_rep) * args.bench_seed_stride
            )
            utils.set_seed(rep_seed)

        prev_n = len(planner.policy.ncp_pred_time_list)
        sync_cuda()
        t0 = time.perf_counter()
        m_out = planner.policy.gen_cond_stgl(g_cond=g_cond, b_s=planner.b_size_per_prob)
        sync_cuda()
        dt = time.perf_counter() - t0

        sample_dt = None
        if len(planner.policy.ncp_pred_time_list) > prev_n:
            sample_dt = float(planner.policy.ncp_pred_time_list[-1][1])

        rec = dict(
            rep_idx=i_rep,
            seed=rep_seed,
            plan_time_s=float(dt),
            sample_time_s=sample_dt,
            plan_len=int(len(m_out.pick_traj)),
            stitch_info=copy.deepcopy(getattr(planner.policy, "last_stitch_info", {})),
        )
        records.append(rec)
        print(
            f"  rep {i_rep:2d}  "
            f"plan={dt:.4f}s"
            + (f"  sample={sample_dt:.4f}s" if sample_dt is not None else "")
        )

    plan_times = np.array([rec["plan_time_s"] for rec in records], dtype=np.float64)
    sample_times = np.array(
        [rec["sample_time_s"] for rec in records if rec["sample_time_s"] is not None],
        dtype=np.float64,
    )

    summary = OrderedDict(
        method_name=args.eval_method,
        meta_method=args.ev_meta_method,
        env=args.dataset,
        seed=pl_seed,
        problem_idx=i_ep,
        n_repeat=int(args.bench_n_repeat),
        n_warmup=int(args.bench_warmup),
        cp_infer_t_type=str(getattr(planner.policy, "cp_infer_t_type", "unknown")),
        n_comp=int(planner.pol_config["ev_n_comp"]),
        b_size_per_prob=int(planner.b_size_per_prob),
        density_p_ratio=float(getattr(args, "ev_density_p_ratio", 0.0)),
        global_density_weight=float(getattr(args, "ev_global_density_weight", 0.0)),
        global_density_inter_rate=int(
            getattr(args, "ev_global_density_inter_rate", 0)
        ),
        global_density_n_mc=int(getattr(args, "ev_global_density_n_mc", 0)),
        global_density_proxy_type=str(
            getattr(args, "ev_global_density_proxy_type", "")
        ),
        global_density_proxy_overlap_weight=float(
            getattr(args, "ev_global_density_proxy_overlap_weight", 0.0)
        ),
        plan_time_mean_s=float(np.mean(plan_times)),
        plan_time_std_s=float(np.std(plan_times)),
        plan_time_min_s=float(np.min(plan_times)),
        plan_time_max_s=float(np.max(plan_times)),
        plan_len=int(records[0]["plan_len"]) if records else 0,
        stitch_info=copy.deepcopy(getattr(planner.policy, "last_stitch_info", {})),
    )
    if len(sample_times) > 0:
        summary["sample_time_mean_s"] = float(np.mean(sample_times))
        summary["sample_time_std_s"] = float(np.std(sample_times))
        summary["sample_time_min_s"] = float(np.min(sample_times))
        summary["sample_time_max_s"] = float(np.max(sample_times))

    os.makedirs(planner.savepath, exist_ok=True)
    summary_path = os.path.join(planner.savepath, "plan_time_summary.json")
    detail_path = os.path.join(planner.savepath, "plan_time_detail.json")
    warmup_path = os.path.join(planner.savepath, "plan_time_warmup.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(detail_path, "w") as f:
        json.dump(records, f, indent=2)
    with open(warmup_path, "w") as f:
        json.dump(warmup_records, f, indent=2)

    print(f"\n[summary] {summary_path}")
    print(f"[detail]  {detail_path}")
    print(f"[warmup]  {warmup_path}")
    print(f"\n{'=' * 60}")
    print(f" Plan Time Summary — {args.dataset} / {args.eval_method}")
    print(f"{'=' * 60}")
    print(
        f"  Full plan time: {summary['plan_time_mean_s']:.4f} ± {summary['plan_time_std_s']:.4f} s"
    )
    if "sample_time_mean_s" in summary:
        print(
            f"  Sample only:    {summary['sample_time_mean_s']:.4f} ± {summary['sample_time_std_s']:.4f} s"
        )
    print(f"  Problem idx:    {i_ep}")
    print(f"  n_comp:         {summary['n_comp']}")
    print(f"  cp_infer_t:     {summary['cp_infer_t_type']}")
    print(f"{'=' * 60}\n")

    planner.env.close()
    return summary


if __name__ == "__main__":
    main()
