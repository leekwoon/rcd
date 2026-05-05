import argparse
import os
import sys

import h5py
import numpy as np

sys.path.append("./")


def make_env(env_name):
    import ogbench

    wrapped_env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = env_name
    return env


def is_xy_in_wall(env, xy):
    i, j = env.xy_to_ij(np.asarray(xy))
    return int(env.maze_map[i, j]) == 1


def generate_humanoid_probs(env_name, n_prob_per_task=20, seed_start=0):
    env = make_env(env_name)
    start_states = []
    goal_states = []

    for task_id in range(1, env.num_tasks + 1):
        for local_idx in range(n_prob_per_task):
            problem_seed = seed_start + (task_id - 1) * n_prob_per_task + local_idx
            env.set_seed_addn(problem_seed)
            _, info = env.reset(options={"task_id": task_id})

            start_state = env.get_state_full().copy()
            goal_state = np.asarray(info["goal_full_jnt_state"]).copy()

            if is_xy_in_wall(env, start_state[:2]):
                raise RuntimeError(
                    f"{env_name}: generated start in wall for task_id={task_id}, "
                    f"local_idx={local_idx}, seed={problem_seed}, xy={start_state[:2]}"
                )
            if is_xy_in_wall(env, goal_state[:2]):
                raise RuntimeError(
                    f"{env_name}: generated goal in wall for task_id={task_id}, "
                    f"local_idx={local_idx}, seed={problem_seed}, xy={goal_state[:2]}"
                )

            start_states.append(start_state)
            goal_states.append(goal_state)

    return {
        "start_state": np.asarray(start_states),
        "goal_pos": np.asarray(goal_states),
    }


def summarize_validity(env_name, data_dict):
    env = make_env(env_name)
    start_in_wall = 0
    goal_in_wall = 0
    for start_state, goal_state in zip(data_dict["start_state"], data_dict["goal_pos"]):
        start_in_wall += int(is_xy_in_wall(env, start_state[:2]))
        goal_in_wall += int(is_xy_in_wall(env, goal_state[:2]))
    return {
        "env_name": env_name,
        "n_probs": int(len(data_dict["start_state"])),
        "start_in_wall": int(start_in_wall),
        "goal_in_wall": int(goal_in_wall),
    }


def default_output_path(env_name, n_prob_per_task, seed_start):
    prefix = {
        "humanoidmaze-giant-stitch-v0": "ogb_HumM_Gi",
        "humanoidmaze-large-stitch-v0": "ogb_HumM_Lg",
        "humanoidmaze-medium-stitch-v0": "ogb_HumM_Me",
    }[env_name]
    return (
        f"data/ogb_maze/ev_probs/"
        f"{prefix}_ev_prob_numEp{n_prob_per_task}_eSdSt{seed_start}_full_jnt_fixed.hdf5"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="+",
        default=[
            "humanoidmaze-giant-stitch-v0",
            "humanoidmaze-large-stitch-v0",
            "humanoidmaze-medium-stitch-v0",
        ],
    )
    parser.add_argument("--n_prob_per_task", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=0)
    args = parser.parse_args()

    for env_name in args.envs:
        out_path = default_output_path(
            env_name=env_name,
            n_prob_per_task=args.n_prob_per_task,
            seed_start=args.seed_start,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data_dict = generate_humanoid_probs(
            env_name=env_name,
            n_prob_per_task=args.n_prob_per_task,
            seed_start=args.seed_start,
        )
        with h5py.File(out_path, "w") as f:
            f.create_dataset("start_state", data=data_dict["start_state"])
            f.create_dataset("goal_pos", data=data_dict["goal_pos"])
            f.attrs["source"] = "generated_from_env_reset"
            f.attrs["n_prob_per_task"] = int(args.n_prob_per_task)
            f.attrs["seed_start"] = int(args.seed_start)
        summary = summarize_validity(env_name, data_dict)
        print(f"[saved] {out_path}")
        print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
