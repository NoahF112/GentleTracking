import argparse
import json
import os
import sys

import torch
import wandb
import hydra
from tqdm import tqdm

from omegaconf import OmegaConf
from isaaclab.app import AppLauncher
from torchrl.envs import ExplorationType, set_exploration_type

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.helpers import make_env_policy


def _unwrap_env(env):
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env
    return base_env


def _calc_keypoint_error(cmd, idx_asset, idx_motion):
    if idx_asset.numel() == 0 or idx_motion.numel() == 0:
        return None
    actual = cmd.asset.data.body_pos_w[:, idx_asset]
    target = cmd.reward_keypoints_w[:, idx_motion]
    diff = target - actual
    return diff.norm(dim=-1).mean(dim=-1)


def _load_run_cfg(run_path, iterations=None):
    api = wandb.Api()
    run = api.run(run_path)
    print(f"Loading run {run.name}")

    root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
    os.makedirs(root, exist_ok=True)

    checkpoints = []
    for file in run.files():
        if "checkpoint" in file.name:
            checkpoints.append(file)
        elif file.name in {"cfg.yaml", "files/cfg.yaml", "config.yaml"}:
            file.download(root, replace=True)

    checkpoint = None
    if iterations is None:
        def sort_by_time(file):
            number_str = file.name[:-3].split("_")[-1]
            if number_str == "final":
                return 100000
            return int(number_str)
        checkpoints.sort(key=sort_by_time)
        checkpoint = checkpoints[-1]
    else:
        for file in checkpoints:
            if file.name == f"checkpoint_{iterations}.pt":
                checkpoint = file
                break
    if checkpoint is None:
        raise RuntimeError("Checkpoint not found.")
    checkpoint.download(root, replace=True)

    try:
        cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
    except FileNotFoundError:
        cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))

    return cfg, os.path.join(root, checkpoint.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-i", "--iterations", dest="iterations", type=int, default=None)
    parser.add_argument("--steps-per-batch", type=int, default=500)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--log-path", type=str, default="outputs/motion_errors.jsonl")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--sequential-start", type=int, default=0)
    parser.add_argument("--no-sequential-wrap", action="store_true", default=False)
    parser.add_argument("--disable-termination", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    cfg, checkpoint_path = _load_run_cfg(args.run_path, iterations=args.iterations)
    OmegaConf.set_struct(cfg, False)
    cfg["checkpoint_path"] = checkpoint_path
    cfg["vecnorm"] = "eval"

    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        cfg["task"]["observation"] = _cfg.task.observation
        cfg["task"]["action"] = _cfg.task.action
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["robot"] = _cfg.task.robot
        cfg["task"]["terrain"] = _cfg.task.terrain
        cfg["task"]["command"] = _cfg.task.command
        cfg["task"]["flags"] = _cfg.task.flags

    if args.num_envs is not None:
        cfg["task"]["num_envs"] = args.num_envs

    cfg["task"]["command"]["dataset"]["sequential_start"] = args.sequential_start
    cfg["task"]["command"]["dataset"]["sequential_wrap"] = not args.no_sequential_wrap
    cfg["task"]["command"]["debug_mode"] = True

    if args.disable_termination:
        cfg["task"]["termination"] = {}

    cfg["app"]["headless"] = bool(args.headless)

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, agent, _, _ = make_env_policy(cfg)
    policy_eval = agent.get_rollout_policy("eval")

    base_env = _unwrap_env(env)
    cmd = base_env.command_manager
    device = base_env.device
    num_envs = base_env.num_envs

    log_path = os.path.abspath(args.log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    lower_sum = torch.zeros(num_envs, device=device)
    upper_sum = torch.zeros(num_envs, device=device)
    lower_max = torch.zeros(num_envs, device=device)
    upper_max = torch.zeros(num_envs, device=device)
    count = torch.zeros(num_envs, dtype=torch.int32, device=device)

    env.eval()
    tensordict_ = env.reset()

    batches = args.batches
    if batches is None:
        if hasattr(cmd.dataset, "_total_motions"):
            total_motions = int(cmd.dataset._total_motions)
            batches = (total_motions + num_envs - 1) // num_envs
        else:
            raise RuntimeError("Auto batch calculation requires a sequential dataset with _total_motions.")

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for batch in tqdm(range(batches), desc="batches"):
            for _ in tqdm(range(args.steps_per_batch), desc=f"batch {batch}", leave=False):
                tensordict_ = policy_eval(tensordict_)
                _, tensordict_ = env.step_and_maybe_reset(tensordict_)

                lower_err = _calc_keypoint_error(cmd, cmd.lower_keypoint_idx_asset, cmd.lower_keypoint_idx_motion)
                if lower_err is not None:
                    lower_sum.add_(lower_err)
                    lower_max = torch.maximum(lower_max, lower_err)
                upper_err = _calc_keypoint_error(cmd, cmd.upper_keypoint_idx_asset, cmd.upper_keypoint_idx_motion)
                if upper_err is not None:
                    upper_sum.add_(upper_err)
                    upper_max = torch.maximum(upper_max, upper_err)
                count.add_(1)

            dataset = cmd.dataset
            if hasattr(dataset, "get_current_motion_ids"):
                motion_ids = dataset.get_current_motion_ids()
            else:
                motion_ids = dataset.motion_ids

            records = []
            for i in range(num_envs):
                steps = int(count[i].item())
                if steps <= 0:
                    continue
                records.append(
                    {
                        "env_id": i,
                        "motion_id": int(motion_ids[i].item()),
                        "steps": steps,
                        "lower_error": float(lower_sum[i].item() / steps) if cmd.lower_keypoint_idx_asset.numel() else None,
                        "lower_max_error": float(lower_max[i].item()) if cmd.lower_keypoint_idx_asset.numel() else None,
                        "upper_error": float(upper_sum[i].item() / steps) if cmd.upper_keypoint_idx_asset.numel() else None,
                        "upper_max_error": float(upper_max[i].item()) if cmd.upper_keypoint_idx_asset.numel() else None,
                    }
                )

            with open(log_path, "a", encoding="utf-8") as f:
                for record in records:
                    record["batch"] = batch
                    record["steps_per_batch"] = args.steps_per_batch
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")

            lower_sum.zero_()
            upper_sum.zero_()
            lower_max.zero_()
            upper_max.zero_()
            count.zero_()
            if hasattr(cmd.dataset, "resample_all"):
                cmd.dataset.resample_all()
            tensordict_ = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
