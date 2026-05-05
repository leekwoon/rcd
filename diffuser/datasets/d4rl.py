import os
import sys
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def _collect_runtime_markers():
    markers = []
    for raw in (
        os.getenv("CONDA_DEFAULT_ENV", ""),
        os.getenv("CONDA_PREFIX", ""),
        sys.executable,
    ):
        if not raw:
            continue
        parts = [raw]
        cur = raw
        for _ in range(3):
            cur = os.path.dirname(cur)
            if not cur:
                break
            parts.append(os.path.basename(cur))
        markers.extend(parts)
    return " ".join(markers).lower()


def is_ogbench_env_name(name):
    if not isinstance(name, str):
        return False
    lname = name.lower()
    return (
        "stitch" in lname
        or "explore" in lname
        or "humanoidmaze" in lname
        or "antsoccer" in lname
    )


_runtime_markers = _collect_runtime_markers()
Is_Gym_Robot_Env = "hi_diffuser_ben" in _runtime_markers
Is_OgB_Robot_Env = (
    "ogb" in _runtime_markers
    or "compdiffuser" in _runtime_markers
)

gym_na = None
minari = None
_HAS_OGBENCH = False
_HAS_D4RL = False

with suppress_output():
    ## d4rl and ogbench register envs on import, so keep them optional and lazy.
    if Is_Gym_Robot_Env:
        ## NOTE: [Only in Eval for Maze2D Ben]
        ## load from gymnasium
        import gymnasium as gym_na
        import minari
        from diffuser.datasets.gym_robo_utils import get_gym_robo_env_name

    try:
        import ogbench  # noqa: F401

        _HAS_OGBENCH = True
    except Exception:
        _HAS_OGBENCH = False

    try:
        import d4rl  # noqa: F401

        _HAS_D4RL = True
    except Exception:
        _HAS_D4RL = False

# -----------------------------------------------------------------------------#
# -------------------------------- general api --------------------------------#
# -----------------------------------------------------------------------------#


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    if is_ogbench_env_name(name):
        if not _HAS_OGBENCH:
            raise ModuleNotFoundError(
                f"OGBench env '{name}' requires the 'ogbench' package, but it is not importable."
            )
    elif not Is_Gym_Robot_Env and not _HAS_D4RL:
        raise ModuleNotFoundError(
            f"Env '{name}' requires the 'd4rl' package, but it is not importable."
        )
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    ## setup the name for choosing env specific value from dict later
    if "hopper" in name or "halfcheetah" in name or "walker" in name:
        ## do not support this envs
        from diffuser.locomo.loco_misc import get_loco_short_env_name

        env.short_name = get_loco_short_env_name(name)

    return env


def load_env_gym_robo(name_ori):
    """
    load a gym robotic env, for Ben
    """
    if type(name_ori) != str:
        ## name is already an environment
        return name_ori

    from diffuser.datasets.gym_robo_utils import get_gym_robo_env_name

    ## get the gym robot version env name to load
    e_name = get_gym_robo_env_name(name_ori)
    with suppress_output():
        is_cont_tk = "maze" not in e_name.lower()  ## False for Maze
        wrapped_env = gym_na.make(
            e_name, continuing_task=is_cont_tk, render_mode="rgb_array"
        )

    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = e_name
    ## important: no noise when reset, noise already added in prob
    # env.position_noise_range = 0.0 ## no need now, we have reset_given
    from diffuser.datasets.comp.d4rl_m2d_const import get_str_maze_spec

    env.str_maze_spec = get_str_maze_spec(e_name)

    return env


def get_dataset(env):
    # dataset = env.get_dataset()
    ## important: Luo update
    h5path = getattr(env, "dset_h5path", None)
    from diffuser.utils import print_color

    if h5path is not None:
        tmp_str = "!" * 50
        print_color(f"\n{tmp_str}\n")
        print_color(f"LuoTest: \n [Loading from LuoTest] {h5path}")
        print_color(f"\n{tmp_str}\n")
        from time import sleep
    print_color(
        f"d4rl.get_dataset: [Loading from LuoTest given h5path] {h5path}", c="y"
    )
    print_color(f"d4rl.get_dataset: {env._dataset_url=}")

    dataset = env.get_dataset(h5path=h5path)

    if "antmaze" in str(env).lower():
        assert False, "not used."
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    # pdb.set_trace()

    return dataset


def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        for k in dataset:
            if "metadata" in k:
                continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            if "maze2d" in env.name and env.proc_m2d_ep:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


# -----------------------------------------------------------------------------#
# -------------------------------- maze2d fixes -------------------------------#
# -----------------------------------------------------------------------------#


def process_maze2d_episode(episode):
    """
    adds in `next_observations` field to episode
    """
    assert "next_observations" not in episode
    length = len(episode["observations"])
    next_observations = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode
