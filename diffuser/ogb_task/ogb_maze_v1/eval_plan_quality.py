"""
Plan Quality Evaluation — No Rollout
=====================================
Generate plans via the diffusion model and evaluate quality by:
  1. Wall penetration rate (what fraction of trajectory points land in walls)
  2. Plan generation time
  3. Visualization of all plans per task overlaid on the maze

Usage:
    python diffuser/ogb_task/ogb_maze_v1/eval_plan_quality.py \
        --config config/ogb_pnt_maze/og_pntM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py \
        --pl_seeds 0
"""

import sys, os

sys.path.append("./")
import time, json, copy, math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import diffuser.utils as utils
from diffuser.guides.render_utils import plot_start_goal


class Parser(utils.Parser):
    dataset: str = None
    config: str = None
    pl_seeds: str = "-1"
    plan_n_ep: int = -100
    ep_per_task: int = -1
    ep_st_idx: int = 0
    save_logbase: str = None
    eval_method: str = "compdiffuser"
    ev_meta_method: str = "baseline"
    ev_density_p_ratio: float = 0.35
    ev_density_n_mc: int = 2
    ev_search_beam_width: int = 6
    ev_search_chunk_pool: int = 4
    ev_search_density_weight: float = 0.35
    ev_search_overlap_weight: float = 1.0
    ev_search_vel_weight: float = 0.35
    ev_search_acc_weight: float = 0.15
    ev_search_rough_weight: float = 0.05
    ev_search_commit_weight: float = 0.0
    ev_search_center_ratio: float = 0.5
    ev_search_edit_weight: float = 0.15
    ev_search_density_gate_temp: float = 0.25
    ev_search_hold_ratio: float = 0.25
    ev_global_overlap_weight: float = 1.0
    ev_risk_threshold: float = 3.0
    ev_switch_margin: float = 0.5
    ev_repair_top_k: int = 2
    ev_local_density_weight: float = 0.0
    ev_local_focus_ratio: float = 0.5
    ev_local_cost_guard: float = 0.0
    ev_local_rank_weight: float = 0.0
    ev_global_density_weight: float = 0.0
    ev_global_density_n_mc: int = 1
    ev_global_density_inter_rate: int = 1
    ev_global_density_t_mid: float = 0.2
    ev_global_density_candidate_topk: int = 8
    ev_global_density_proxy_type: str = "window_mean"
    ev_global_density_proxy_overlap_weight: float = 0.0
    ev_global_density_window_beta: float = 1.0
    pq_logbase: str = "logs_pq"
    time_logbase: str = "logs_plan_time"
    bench_n_repeat: int = 20
    bench_warmup: int = 2
    bench_problem_idx: int = 0
    bench_seed_stride: int = 1

    def mkdir(self, args):
        # eval-only entrypoint: do not touch the original checkpoint log tree
        return

    def save_diff(self, args):
        return


def apply_eval_method_alias(args):
    if getattr(args, "ev_meta_method", "baseline") != "baseline":
        return
    method = getattr(args, "eval_method", "compdiffuser")
    if method == "rcd":
        args.ev_meta_method = "rcd"
    elif method == "cdgs":
        args.ev_meta_method = "cdgs"


def apply_rcd_defaults(args):
    if getattr(args, "ev_meta_method", None) != "rcd":
        return
    # RCD uses the overlap-aware coupled proxy by default.
    if float(getattr(args, "ev_global_density_proxy_overlap_weight", 0.0)) == 0.0:
        args.ev_global_density_proxy_overlap_weight = 0.5


# ── maze helpers ──────────────────────────────────────────────


def parse_maze_grid(env_name: str):
    """Return 2-D list-of-strings grid (borders stripped) for a given OGBench env."""
    from ogbench.luo_utils.d4rl_m2d_const import get_str_maze_spec

    maze_string = get_str_maze_spec(env_name)
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def ij_point_in_wall(ij_pt, maze_grid):
    """Check if a single (i, j) point lies inside a wall cell."""
    gi = int(np.floor(ij_pt[0] - 0.5))
    gj = int(np.floor(ij_pt[1] - 0.5))
    n_rows = len(maze_grid)
    n_cols = len(maze_grid[0]) if n_rows > 0 else 0
    if gi < 0 or gi >= n_rows or gj < 0 or gj >= n_cols:
        return True  # out of bounds → wall
    return maze_grid[gi][gj] == "#"


def wall_penetration_stats(ij_traj, maze_grid):
    """
    Given a trajectory in ij coords (H, 2), compute wall penetration statistics.
    Returns dict with n_points, n_wall, wall_ratio, wall_indices.
    """
    n_pts = len(ij_traj)
    wall_mask = np.array(
        [ij_point_in_wall(ij_traj[t], maze_grid) for t in range(n_pts)]
    )
    return dict(
        n_points=int(n_pts),
        n_wall=int(wall_mask.sum()),
        wall_ratio=float(wall_mask.mean()),
        wall_indices=np.where(wall_mask)[0].tolist(),
    )


def combine_wall_stats(stats_list):
    if len(stats_list) == 1:
        return stats_list[0]
    n_points = int(sum(stat["n_points"] for stat in stats_list))
    n_wall = int(sum(stat["n_wall"] for stat in stats_list))
    wall_indices = {
        int(idx)
        for stat in stats_list
        for idx in stat.get("wall_indices", [])
    }
    return dict(
        n_points=n_points,
        n_wall=n_wall,
        wall_ratio=float(n_wall / max(n_points, 1)),
        wall_indices=sorted(wall_indices),
    )


def get_plan_quality_tracks(dataset_name, pick_traj, start_state, goal_pos):
    """
    Decide which 2-D projections to check for wall penetration.
    For AntSoccer we conservatively check both ant position and ball position.
    """
    tracks = [
        dict(
            name="agent",
            traj_xy=np.asarray(pick_traj[:, :2]),
            start_xy=np.asarray(start_state[:2]),
            goal_xy=np.asarray(goal_pos[:2]),
        )
    ]
    if "antsoccer" in dataset_name.lower() and pick_traj.shape[1] >= 4:
        tracks.append(
            dict(
                name="ball",
                traj_xy=np.asarray(pick_traj[:, -2:]),
                start_xy=np.asarray(start_state[-2:]),
                goal_xy=np.asarray(goal_pos[-2:]),
            )
        )
    return tracks


# ── plotting ──────────────────────────────────────────────────


def plot_maze_layout(ax, maze_grid):
    """Draw maze walls (reuses logic from render_utils.py)."""
    ax.clear()
    for i, row in enumerate(maze_grid):
        for j, cell in enumerate(row):
            if cell == "#":
                sq = plt.Rectangle(
                    (i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black"
                )
                ax.add_patch(sq)
    ax.set_aspect("equal")
    n_rows = len(maze_grid)
    n_cols = len(maze_grid[0])
    ax.set_xlim(0.5, n_rows + 0.5)
    ax.set_ylim(0.5, n_cols + 0.5)
    ax.set_xticks(np.arange(0.5, n_rows + 0.5))
    ax.set_yticks(np.arange(0.5, n_cols + 0.5))
    ax.grid(True, color="white", linewidth=2)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    for sp in ax.spines.values():
        sp.set_linewidth(2)


def plot_plans_on_maze(
    maze_grid, ij_trajs, start_ij, goal_ij, wall_stats_list, title="", save_path=None
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    plot_maze_layout(ax, maze_grid)

    import matplotlib.colors as mcolors

    for idx, (traj, ws) in enumerate(zip(ij_trajs, wall_stats_list)):
        n = len(traj)
        has_wall = ws["n_wall"] > 0
        base_cmap = plt.cm.Reds if has_wall else plt.cm.Blues
        # t ∈ [0,1] along trajectory; shift colormap range to [0.25, 0.85] to avoid too-faint starts
        t = np.linspace(0.25, 0.85, n)
        colors = base_cmap(t)
        colors[:, 3] = np.linspace(0.3, 0.9, n)
        ax.scatter(traj[:, 0], traj[:, 1], c=colors, s=4, zorder=3)

    # Match rollout rendering: explicit start/goal markers on top of overlaid plans.
    plot_start_goal(ax, (np.asarray(start_ij), np.asarray(goal_ij)))

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[saved] {save_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────


def setup_args():
    """Replicates the arg setup from plan_ogb_stgl_sml.py __main__ block."""
    from diffuser.datasets.d4rl import Is_OgB_Robot_Env

    assert Is_OgB_Robot_Env

    args_train = Parser().parse_args("diffusion")
    args = Parser().parse_args("plan")
    apply_eval_method_alias(args)
    apply_rcd_defaults(args)

    loadpath = args.logbase, args.dataset, args_train.exp_name
    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds)
    args.n_batch_acc_probs = 4

    args.is_replan = None
    args.n_act_per_waypnt = 2
    args.is_save_pkl = False
    args.is_rd_agv = False

    dfu_ndim = len(args_train.dataset_config["obs_select_dim"])

    repl_wp_cfg = {}

    if "pointmaze" in args.dataset.lower():
        if "giant" in args.dataset:
            args.ev_n_comp = 8
            args.ev_cp_infer_t_type = "interleave"
            args.is_replan = "ada_dist"
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                thres=1,
                type="m_2",
                ada_dist_minus_n_wp=10,
                cond_2_extra=150,
                n_max_steps=1000,
            )
            args.inv_epoch = int(8e5)
        elif "large" in args.dataset:
            args.ev_n_comp = 6
            args.ev_cp_infer_t_type = "interleave"
            args.is_replan = "ada_dist"
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                max_n_repl=0,
                thres=1,
                type="m_2",
                ada_dist_minus_n_wp=0,
                cond_2_extra=150,
                n_max_steps=1000,
            )
            args.inv_epoch = int(8e5)
        elif "medium" in args.dataset:
            args.ev_n_comp = 3
            args.ev_cp_infer_t_type = "interleave"
            args.is_replan = "ada_dist"
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                max_n_repl=0,
                thres=1,
                type="m_2",
                ada_dist_minus_n_wp=0,
                cond_2_extra=150,
                n_max_steps=1000,
            )
            args.inv_epoch = int(8e5)
    elif "antmaze" in args.dataset.lower():
        args.ev_cp_infer_t_type = "interleave"
        args.is_replan = "ada_dist"
        args.n_act_per_waypnt = 1
        args.inv_epoch = int(8e5)
        if "giant" in args.dataset:
            args.ev_n_comp = 9 if dfu_ndim == 2 else 9
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                thres=4,
                type="m_2",
                ada_dist_minus_n_wp=50,
                cond_2_extra=150,
                n_max_steps=2000,
            )
            if dfu_ndim > 2:
                args.repl_ada_dist_cfg["ada_dist_minus_n_wp"] = 0
                args.repl_ada_dist_cfg["max_n_repl"] = 15
                args.repl_ada_dist_cfg["used_idxs"] = (0, 1)
        elif "large" in args.dataset:
            args.ev_n_comp = 5 if dfu_ndim <= 2 else (6 if dfu_ndim == 15 else 5)
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                thres=4,
                type="m_2",
                ada_dist_minus_n_wp=50,
                cond_2_extra=150,
                n_max_steps=1000,
                used_idxs=(0, 1),
            )
        elif "medium" in args.dataset:
            args.ev_n_comp = 3
            args.repl_ada_dist_cfg = dict(
                max_n_repl=0,
                thres=4,
                type="m_2",
                ada_dist_minus_n_wp=0,
                cond_2_extra=150,
                n_max_steps=1000,
            )
    elif "antsoccer" in args.dataset.lower():
        args.ev_cp_infer_t_type = "interleave"
        args.is_replan = "ada_dist"
        args.inv_epoch = "latest"
        args.is_inv_train_mode = True
        if "arena" in args.dataset:
            if dfu_ndim == 17:
                args.ev_n_comp = 5
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type="m_2",
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50,
                    n_max_steps=5000,
                    used_idxs=(0, 1),
                )
            elif dfu_ndim == 4:
                args.ev_n_comp = 7
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type="m_2",
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50,
                    n_max_steps=5000,
                    used_idxs=(0, 1),
                )
            else:
                raise NotImplementedError(
                    f"Unsupported AntSoccer arena obs dim: {dfu_ndim}"
                )
        elif "medium" in args.dataset:
            if dfu_ndim == 17:
                args.ev_n_comp = 6
                args.n_act_per_waypnt = 2
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=6,
                    type="m_2",
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=10,
                    n_max_steps=5000,
                    used_idxs=(0, 1),
                )
            elif dfu_ndim == 4:
                args.ev_n_comp = 8
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type="m_2",
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50,
                    n_max_steps=5000,
                    used_idxs=(0, 1),
                )
            else:
                raise NotImplementedError(
                    f"Unsupported AntSoccer medium obs dim: {dfu_ndim}"
                )
        else:
            raise NotImplementedError(f"Unsupported AntSoccer env: {args.dataset}")
    else:
        raise NotImplementedError(f"Unsupported env: {args.dataset}")

    args.b_size_per_prob = 40
    args.ev_top_n = 5
    args.ev_pick_type = "first"
    args.tjb_blend_type = "exp"
    args.tjb_exp_beta = 2
    args.ev_meta_method = getattr(args, "ev_meta_method", "baseline")
    args.ev_density_p_ratio = float(getattr(args, "ev_density_p_ratio", 0.35))
    args.ev_density_n_mc = int(getattr(args, "ev_density_n_mc", 2))
    args.ev_search_beam_width = int(getattr(args, "ev_search_beam_width", 6))
    args.ev_search_chunk_pool = int(getattr(args, "ev_search_chunk_pool", 4))
    args.ev_search_density_weight = float(
        getattr(args, "ev_search_density_weight", 0.35)
    )
    args.ev_search_overlap_weight = float(
        getattr(args, "ev_search_overlap_weight", 1.0)
    )
    args.ev_search_vel_weight = float(getattr(args, "ev_search_vel_weight", 0.35))
    args.ev_search_acc_weight = float(getattr(args, "ev_search_acc_weight", 0.15))
    args.ev_search_rough_weight = float(
        getattr(args, "ev_search_rough_weight", 0.05)
    )
    args.ev_search_density_gate_temp = float(
        getattr(args, "ev_search_density_gate_temp", 0.25)
    )
    args.ev_global_overlap_weight = float(
        getattr(args, "ev_global_overlap_weight", 1.0)
    )
    args.ev_risk_threshold = float(getattr(args, "ev_risk_threshold", 3.0))
    args.ev_switch_margin = float(getattr(args, "ev_switch_margin", 0.5))
    args.ev_global_density_weight = float(
        getattr(args, "ev_global_density_weight", 0.0)
    )
    args.ev_global_density_n_mc = int(getattr(args, "ev_global_density_n_mc", 1))
    args.ev_global_density_inter_rate = int(
        getattr(args, "ev_global_density_inter_rate", 1)
    )
    args.ev_global_density_t_mid = float(
        getattr(args, "ev_global_density_t_mid", 0.2)
    )
    args.ev_global_density_candidate_topk = int(
        getattr(args, "ev_global_density_candidate_topk", 8)
    )
    args.ev_global_density_proxy_type = str(
        getattr(args, "ev_global_density_proxy_type", "window_mean")
    )
    args.ev_global_density_proxy_overlap_weight = float(
        getattr(args, "ev_global_density_proxy_overlap_weight", 0.0)
    )
    args.ev_global_density_window_beta = float(
        getattr(args, "ev_global_density_window_beta", 1.0)
    )
    apply_rcd_defaults(args)
    args.var_temp = 1.0
    args.cond_w = 2.0
    args.use_ddim = True
    args.ddim_eta = 1.0
    args.ddim_steps = 50
    args.repl_wp_cfg = repl_wp_cfg

    if args.is_replan == "ada_dist":
        args.env_n_max_steps = args.repl_ada_dist_cfg["n_max_steps"]
    else:
        args.env_n_max_steps = None

    latest_e = utils.get_latest_epoch(loadpath)
    args_train.diffusion_epoch = latest_e
    args.diffusion_epoch = latest_e

    from datetime import datetime

    sub_dir = (
        f"{datetime.now().strftime('%y%m%d-%H%M%S-%f')[:-3]}"
        f"-{args.eval_method}"
        f"-pq-ncp{args.ev_n_comp}"
        f"-evSd{','.join(str(s) for s in args.pl_seeds)}"
    )
    args.savepath = os.path.join(
        args.pq_logbase, args.dataset, f"plan_quality_{args.eval_method}", sub_dir
    )

    return args_train, args, loadpath


def main():
    args_train, args, loadpath = setup_args()

    pl_seed = args.pl_seeds[0]
    if pl_seed == -1:
        pl_seed = None

    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import (
        OgB_Stgl_Sml_MazeEnvPlanner_V1,
    )

    planner = OgB_Stgl_Sml_MazeEnvPlanner_V1(
        copy.deepcopy(args_train), args=copy.deepcopy(args)
    )
    planner.setup_load(ld_config={})

    env = planner.env
    env_name = planner.args.dataset
    obs_sel = planner.obs_select_dim
    n_comp = planner.pol_config["ev_n_comp"]

    # ---- maze grid for wall check ----
    maze_grid = parse_maze_grid(env_name)

    # ---- coord converter ----
    from diffuser.datasets.ogb_dset.ogb_utils import ogb_xy_to_ij

    # ---- output dir ----
    seed_tag = f"sd{pl_seed}" if pl_seed is not None else "nosd"
    out_dir = os.path.join(
        planner.savepath_root,
        f"plan_quality_{seed_tag}_{time.strftime('%y%m%d-%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ---- eval problems ----
    total_probs = len(planner.problems_dict["start_state"])
    n_tasks = 5
    n_ep_per_task = total_probs // n_tasks
    if args.ep_per_task > 0:
        n_eval_per_task = min(args.ep_per_task, n_ep_per_task)
    elif args.plan_n_ep > 0:
        n_eval_per_task = min(args.plan_n_ep, n_ep_per_task)
    else:
        n_eval_per_task = n_ep_per_task

    eval_indices = []
    for tk in range(n_tasks):
        start = tk * n_ep_per_task
        eval_indices.extend(range(start, start + n_eval_per_task))
    num_probs = len(eval_indices)

    if pl_seed is not None:
        utils.set_seed(pl_seed)

    print(
        f"\n[eval_plan_quality] method={args.eval_method}  env={env_name}  seed={pl_seed}  "
        f"n_tasks={n_tasks}  ep_per_task={n_eval_per_task}  total={num_probs}"
    )
    print(f"[eval_plan_quality] output → {out_dir}\n")

    all_results = []
    task_trajs = {}
    task_sg = {}

    for i_ep in eval_indices:
        task_idx = i_ep // n_ep_per_task

        st_state = planner.problems_dict["start_state"][i_ep]
        gl_pos = planner.problems_dict["goal_pos"][i_ep]

        # diffusion input: only the selected obs dims
        input_st = st_state[list(obs_sel)]
        gl_for_dfu = gl_pos[list(obs_sel)]

        g_cond = {
            "st_gl": np.array([input_st[None,], gl_for_dfu[None,]], dtype=np.float32),
        }

        # ---- generate plan ----
        t0 = time.time()
        m_out = planner.policy.gen_cond_stgl(g_cond=g_cond, b_s=planner.b_size_per_prob)
        dt = time.time() - t0

        pick_traj = m_out.pick_traj  # (H, dim), unnormalized mujoco xy

        # ---- convert to ij for wall check ----
        track_specs = get_plan_quality_tracks(env_name, pick_traj, st_state, gl_pos)
        track_results = []
        for track in track_specs:
            traj_ij = ogb_xy_to_ij(env, track["traj_xy"])
            st_ij = ogb_xy_to_ij(env, track["start_xy"].reshape(1, 2))[0]
            gl_ij = ogb_xy_to_ij(env, track["goal_xy"].reshape(1, 2))[0]
            track_results.append(
                dict(
                    name=track["name"],
                    traj_ij=traj_ij,
                    start_ij=st_ij,
                    goal_ij=gl_ij,
                    wall_stats=wall_penetration_stats(traj_ij, maze_grid),
                )
            )

        primary_track = track_results[0]
        ws = combine_wall_stats([track["wall_stats"] for track in track_results])
        st_ij = primary_track["start_ij"]
        gl_ij = primary_track["goal_ij"]
        goal_dist = float(np.linalg.norm(primary_track["traj_ij"][-1] - gl_ij))

        ep_result = dict(
            i_ep=i_ep,
            task=task_idx,
            plan_time_s=round(dt, 3),
            plan_len=len(pick_traj),
            wall_ratio=round(ws["wall_ratio"], 4),
            n_wall=ws["n_wall"],
            n_points=ws["n_points"],
            goal_dist_ij=round(goal_dist, 3),
            wall_tracks=[
                dict(
                    name=track["name"],
                    wall_ratio=round(track["wall_stats"]["wall_ratio"], 4),
                    n_wall=int(track["wall_stats"]["n_wall"]),
                    n_points=int(track["wall_stats"]["n_points"]),
                )
                for track in track_results
            ],
            stitch_info=copy.deepcopy(
                getattr(planner.policy, "last_stitch_info", {})
            ),
        )
        all_results.append(ep_result)

        tag = "OK" if ws["n_wall"] == 0 else f"WALL({ws['n_wall']})"
        print(
            f"  ep {i_ep:3d}  task {task_idx}  "
            f"wall={ws['wall_ratio']:5.1%}  "
            f"goal_d={goal_dist:.2f}  "
            f"t={dt:.2f}s  {tag}"
        )

        # accumulate per task
        task_trajs.setdefault(task_idx, []).append((primary_track["traj_ij"], ws))
        if task_idx not in task_sg:
            task_sg[task_idx] = (st_ij, gl_ij)

    # ---- per-task visualization ----
    print("\n[eval_plan_quality] Generating per-task plan overlay figures...")
    for tk, trajs_ws in task_trajs.items():
        ij_list = [tw[0] for tw in trajs_ws]
        ws_list = [tw[1] for tw in trajs_ws]
        st_ij, gl_ij = task_sg[tk]

        n_wall_total = sum(w["n_wall"] for w in ws_list)
        n_pts_total = sum(w["n_points"] for w in ws_list)
        ratio = n_wall_total / max(n_pts_total, 1)
        title = (
            f"{env_name}  task{tk}  "
            f"wall={ratio:.1%} ({n_wall_total}/{n_pts_total})  "
            f"n_plans={len(ij_list)}"
        )

        save_path = os.path.join(out_dir, f"task{tk}_plans.png")
        plot_plans_on_maze(
            maze_grid, ij_list, st_ij, gl_ij, ws_list, title=title, save_path=save_path
        )

    # ---- summary ----
    wall_ratios = [r["wall_ratio"] for r in all_results]
    times = [r["plan_time_s"] for r in all_results]
    goal_dists = [r["goal_dist_ij"] for r in all_results]

    summary = OrderedDict(
        method_name=args.eval_method,
        meta_method=args.ev_meta_method,
        env=env_name,
        seed=pl_seed,
        n_problems=num_probs,
        wall_ratio_mean=round(float(np.mean(wall_ratios)), 4),
        wall_ratio_std=round(float(np.std(wall_ratios)), 4),
        wall_ratio_max=round(float(np.max(wall_ratios)), 4),
        n_clean_plans=int(sum(1 for r in wall_ratios if r == 0.0)),
        plan_time_mean_s=round(float(np.mean(times)), 3),
        plan_time_std_s=round(float(np.std(times)), 3),
        goal_dist_mean=round(float(np.mean(goal_dists)), 3),
        goal_dist_std=round(float(np.std(goal_dists)), 3),
    )
    summary["stitch_info"] = getattr(planner.policy, "last_stitch_info", {})

    # per-task summary
    per_task = {}
    for tk, trajs_ws in task_trajs.items():
        ws_list = [tw[1] for tw in trajs_ws]
        tk_wall = [w["wall_ratio"] for w in ws_list]
        tk_results = [r for r in all_results if r["task"] == tk]
        tk_times = [r["plan_time_s"] for r in tk_results]
        tk_gdist = [r["goal_dist_ij"] for r in tk_results]
        per_task[f"task{tk}"] = dict(
            n_eps=len(ws_list),
            wall_ratio_mean=round(float(np.mean(tk_wall)), 4),
            n_clean=int(sum(1 for w in tk_wall if w == 0.0)),
            plan_time_mean_s=round(float(np.mean(tk_times)), 3),
            goal_dist_mean=round(float(np.mean(tk_gdist)), 3),
        )
    summary["per_task"] = per_task

    # ---- save ----
    json_path = os.path.join(out_dir, "plan_quality_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] {json_path}")

    detail_path = os.path.join(out_dir, "plan_quality_detail.json")
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[detail]  {detail_path}")

    # print summary
    print(f"\n{'=' * 60}")
    print(f" Plan Quality Summary — {env_name} (seed={pl_seed})")
    print(f"{'=' * 60}")
    print(
        f"  Wall ratio:  {summary['wall_ratio_mean']:.2%} ± {summary['wall_ratio_std']:.2%}  "
        f"(max {summary['wall_ratio_max']:.2%})"
    )
    print(f"  Clean plans: {summary['n_clean_plans']}/{num_probs}")
    print(
        f"  Plan time:   {summary['plan_time_mean_s']:.2f} ± {summary['plan_time_std_s']:.2f} s"
    )
    print(
        f"  Goal dist:   {summary['goal_dist_mean']:.2f} ± {summary['goal_dist_std']:.2f} (ij)"
    )
    print(f"{'=' * 60}\n")

    planner.env.close()
    return summary


if __name__ == "__main__":
    main()
