import argparse
import json
import os
import re
import sys
import time
import select

import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from tqdm import tqdm

from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from active_adaptation.utils.motion import MotionDataset


def _load_cfg(task: str | None, cfg_path: str | None):
    if cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.set_struct(cfg, False)
        return cfg
    if task is None:
        raise ValueError("Either --task or --cfg must be provided.")
    with hydra.initialize(config_path="../cfg", job_name="review", version_base=None):
        cfg = hydra.compose(config_name="eval", overrides=[f"task={task}"])
    OmegaConf.set_struct(cfg, False)
    return cfg


def _find_root_index(body_names):
    root_tokens = ["pelvis", "torso", "waist", "root"]
    for token in root_tokens:
        for i, name in enumerate(body_names):
            if token in name.lower():
                return i
    return None


def _get_keypoint_indices(body_names, patterns):
    if not patterns:
        return list(range(len(body_names)))
    idx = []
    for i, name in enumerate(body_names):
        if any(re.match(p, name) for p in patterns):
            idx.append(i)
    if not idx:
        idx = list(range(len(body_names)))
    root_idx = _find_root_index(body_names)
    if root_idx is not None and root_idx not in idx:
        idx.append(root_idx)
    return idx


def _load_candidates(errors_path, upper_thres, lower_thres):
    motion_ids = []
    seen = set()
    with open(errors_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            upper = rec.get("upper_error")
            lower = rec.get("lower_error")
            upper_ok = isinstance(upper, (int, float)) and upper > upper_thres
            lower_ok = isinstance(lower, (int, float)) and lower > lower_thres
            if not (upper_ok or lower_ok):
                continue
            motion_id = rec.get("motion_id")
            if motion_id is None or motion_id in seen:
                continue
            seen.add(motion_id)
            motion_ids.append(int(motion_id))
    return motion_ids


def _read_stdin_action():
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except Exception:
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    key = line.strip().lower()
    if not key:
        return None
    return key[0]


def _load_existing_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            motion_id = rec.get("motion_id")
            if motion_id is None:
                continue
            status = rec.get("status")
            if status is None:
                if "is_bad" in rec:
                    status = "bad" if rec["is_bad"] else "good"
                elif "label" in rec:
                    status = rec["label"]
                else:
                    status = "unknown"
            labels[int(motion_id)] = status
    return labels


def _record_motion(path, motion_id, status):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"motion_id": motion_id, "status": status}, ensure_ascii=True) + "\n")


def _setup_axes(ax, points):
    mins = points.min(axis=(0, 1))
    maxs = points.max(axis=(0, 1))
    center = (mins + maxs) * 0.5
    radius = max(maxs - mins) * 0.6 + 1e-3

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _build_connections(keypoint_names):
    parts = {"left": {}, "right": {}, "center": {}}
    for i, name in enumerate(keypoint_names):
        lower = name.lower()
        if "left" in lower:
            side = "left"
        elif "right" in lower:
            side = "right"
        else:
            side = "center"

        part = None
        if "head" in lower:
            part = "head"
        elif "shoulder" in lower:
            part = "shoulder"
        elif "wrist" in lower:
            part = "wrist"
        elif "hand" in lower:
            part = "hand"
        elif "knee" in lower:
            part = "knee"
        elif "ankle" in lower:
            part = "ankle"
        elif "pelvis" in lower or "torso" in lower or "root" in lower or "waist" in lower:
            part = "root"

        if part and part not in parts[side]:
            parts[side][part] = i

    def get(side, part):
        return parts.get(side, {}).get(part)

    connections = []
    seen = set()

    def add(a, b):
        if a is None or b is None or a == b:
            return
        pair = (a, b) if a < b else (b, a)
        if pair in seen:
            return
        seen.add(pair)
        connections.append((a, b))

    ls = get("left", "shoulder")
    lw = get("left", "wrist")
    lh = get("left", "hand")
    add(ls, lw if lw is not None else lh)
    add(lw, lh)

    rs = get("right", "shoulder")
    rw = get("right", "wrist")
    rh = get("right", "hand")
    add(rs, rw if rw is not None else rh)
    add(rw, rh)

    lk = get("left", "knee")
    la = get("left", "ankle")
    rk = get("right", "knee")
    ra = get("right", "ankle")

    root = get("center", "root")
    head = get("center", "head") or get("left", "head") or get("right", "head")

    if root is not None:
        add(root, lk)
    add(lk, la)
    if root is not None:
        add(root, rk)
    add(rk, ra)

    def dense(nodes):
        nodes = [n for n in nodes if n is not None]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                add(nodes[i], nodes[j])

    add(root, ls)
    add(root, rs)
    add(root, head)

    return connections


def _play_motion(ax, scatter, lines, connections, points, fps, speed, title, control):
    dt = 1.0 / (fps * speed)
    ax.set_title(title)
    control["action"] = None
    for frame in range(points.shape[0]):
        if not plt.fignum_exists(ax.figure.number):
            return None
        xs = points[frame, :, 0]
        ys = points[frame, :, 1]
        zs = points[frame, :, 2]
        scatter._offsets3d = (xs, ys, zs)
        for line, (a, b) in zip(lines, connections):
            line.set_data([xs[a], xs[b]], [ys[a], ys[b]])
            line.set_3d_properties([zs[a], zs[b]])
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()
        t0 = time.perf_counter()
        while True:
            plt.pause(0.001)
            action = control.get("action") or _read_stdin_action()
            if action:
                return action
            if (time.perf_counter() - t0) >= dt:
                break
        action = control.get("action")
        if action:
            return action
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors-path", required=True, help="JSONL file from eval_motion_errors.py")
    parser.add_argument("--mem-path", required=True, help="Memmap dataset path name (single dataset)")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--output", default="outputs/flagged_motion_ids.jsonl")
    parser.add_argument("--upper-thres", type=float, default=0.1)
    parser.add_argument("--lower-thres", type=float, default=0.2)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--speed", type=float, default=2.0)
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be positive.")

    cfg = _load_cfg(args.task, args.cfg)
    keypoint_patterns = cfg.task.command.get("keypoint_patterns", [])

    ds = MotionDataset.create_from_path_lazy(args.mem_path, device=torch.device("cpu"))
    keypoint_idx = _get_keypoint_indices(ds.body_names, keypoint_patterns)
    keypoint_names = [ds.body_names[i] for i in keypoint_idx]
    connections = _build_connections(keypoint_names)

    existing_labels = _load_existing_labels(args.output)
    candidate_ids = _load_candidates(args.errors_path, args.upper_thres, args.lower_thres)
    candidate_ids = [mid for mid in candidate_ids if mid not in existing_labels]
    if not candidate_ids:
        print("No motions matched the thresholds.")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter([], [], [], s=20)
    lines = [
        ax.plot([], [], [], color="gray", linewidth=1.0, alpha=0.8)[0]
        for _ in connections
    ]
    control = {"action": None}

    def on_key(event):
        key = (event.key or "").lower()
        if key in {"y", "n", "r", "q"}:
            control["action"] = key

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Controls: y=mark bad, n=mark good, r=replay, q=quit")
    print("You can press keys in the figure window or type y/n/r/q + Enter in the terminal.")

    for motion_id in tqdm(candidate_ids, desc="motions"):
        start = int(ds.starts[motion_id])
        end = int(ds.ends[motion_id])
        body_pos = ds.data.body_pos_w[start:end].to(torch.float32).cpu().numpy()
        points = body_pos[:, keypoint_idx, :]
        _setup_axes(ax, points)

        title = f"motion {motion_id} ({end - start} frames)"
        while True:
            action = _play_motion(ax, scatter, lines, connections, points, args.fps, args.speed, title, control)
            if action is None and not plt.fignum_exists(fig.number):
                return
            if action == "q":
                return
            if action == "r":
                continue
            if action == "y":
                _record_motion(args.output, motion_id, "bad")
                break
            if action == "n":
                _record_motion(args.output, motion_id, "good")
                break

            resp = input(
                f"Mark motion {motion_id} as abnormal? [y/N/r/q]: "
            ).strip().lower()
            if resp == "q":
                return
            if resp in {"r", "replay"}:
                continue
            if resp in {"y", "yes"}:
                _record_motion(args.output, motion_id, "bad")
            else:
                _record_motion(args.output, motion_id, "good")
            break

    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    main()
