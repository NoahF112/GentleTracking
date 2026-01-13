import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp import reward, termination, observation
from active_adaptation.utils.multimotion import ProgressiveMultiMotionDataset
from active_adaptation.utils.simple_multimotion import SimpleSequentialMultiMotionDataset
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils.math import (
    quat_apply_inverse,
    quat_apply,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    quat_from_angle_axis,
    yaw_quat,
    matrix_from_quat
)
from .base import Command
import re
import math
import gc
from typing import Sequence


def _match_indices(motion_names, asset_names, patterns, name_map=None, device=None, debug=False):
    asset_idx, motion_idx = [], []
    for i, a in enumerate(asset_names):
        if any(re.match(p, a) for p in patterns):
            m = name_map.get(a, a) if name_map else a
            if m in motion_names:
                asset_idx.append(i)
                motion_idx.append(motion_names.index(m))
                if debug:
                    print(f"Matched asset '{a}' (idx {i}) to motion '{m}' (idx {motion_names.index(m)})")
    return torch.tensor(motion_idx, device=device), torch.tensor(asset_idx, device=device)

def _calc_exp_sigma(error : torch.Tensor, sigma_list : list[float], reduce_last_dim : bool = False):
    count = len(sigma_list)
    if reduce_last_dim:
        rewards = [torch.exp(- error / sigma).mean(dim=-1, keepdim=True) for sigma in sigma_list]
    else:
        rewards = [torch.exp(- error / sigma) for sigma in sigma_list]
    return sum(rewards) / count

def get_items_by_index(list, indexes):
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.tolist()
    return [list[i] for i in indexes]

def convert_dtype(dtype_str):
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': bool,
        'long': torch.long
    }
    if isinstance(dtype_str, str):
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")
        return dtype_map[dtype_str]
    return dtype_str

class MotionTrackingCommand(Command):
    def __init__(self, env, dataset: dict,
                dataset_extra_keys: list[dict] = [],
                keypoint_map: dict = {},
                keypoint_patterns: list[str] = [],
                lower_keypoint_patterns: list[str] = [],
                upper_keypoint_patterns: list[str] = [],
                joint_patterns: list[str] = [],
                ignore_joint_patterns: list[str] = [],
                feet_patterns: list[str] = [],
                init_noise: dict[str, float] = {},
                reward_sigma: dict[str, list[float]] = {},
                student_train: bool = False,
                future_steps: list[int] = [],
                cum_root_pos_scale: float = 0.0,
                cum_keypoint_scale: float = 0.0,
                cum_orientation_scale: float = 0.0,
                boot_indicator_max: int = 0,
                body_z_terminate_thres: float = 0.0,
                body_z_terminate_patterns: list[str] = [],
                reinit_prob: float = 0.0,
                reinit_min_steps: int = 0,
                reinit_max_steps: int = 0,
                gravity_terminate_thres: float = 0.0,
                debug_mode: bool = False,):
        super().__init__(env)
        
        self.future_steps = torch.tensor(future_steps, device=self.device)

        self.zero_init_prob = 1.0

        dataset_extra_keys = [
            {**k, 'dtype': convert_dtype(k['dtype'])} 
            for k in dataset_extra_keys
        ]

        self.student_train = student_train
        self.debug_mode = debug_mode

        dataset_cls = SimpleSequentialMultiMotionDataset if self.debug_mode else ProgressiveMultiMotionDataset
        self.dataset = dataset_cls(
            **dataset,
            env_size=self.num_envs,
            max_step_size=1000,
            dataset_extra_keys=dataset_extra_keys,
            device=self.device,
            ds_device=torch.device("cpu"),
            refresh_threshold=1000 * 20,
        )
        self.dataset.set_limit(self.asset.data.soft_joint_pos_limits, self.asset.data.soft_joint_vel_limits, self.asset.joint_names)

        # bodies for full‑body keypoint tracking
        self.keypoint_patterns = keypoint_patterns
        self.lower_keypoint_patterns = lower_keypoint_patterns
        self.upper_keypoint_patterns = upper_keypoint_patterns
        self.keypoint_map = keypoint_map
        self.keypoint_idx_motion, self.keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )
        self.lower_keypoint_idx_motion, self.lower_keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.lower_keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )
        self.upper_keypoint_idx_motion, self.upper_keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.upper_keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )

        # joints for full‑body joint tracking
        self.joint_patterns = joint_patterns
        self.joint_idx_motion, self.joint_idx_asset = _match_indices(
            self.dataset.joint_names,
            self.asset.joint_names,
            self.joint_patterns,
            device=self.device
        )
        
        self.feet_patterns = feet_patterns
        self.feet_idx_motion, self.feet_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.feet_patterns,
            device=self.device
        )

        # all joints except ankles
        self.ignore_joint_patterns = ignore_joint_patterns
        all_j_m, all_j_a = [], []
        for j in self.asset.joint_names:
            if j in self.dataset.joint_names and not any(re.match(p, j) for p in self.ignore_joint_patterns):
                all_j_m.append(self.dataset.joint_names.index(j))
                all_j_a.append(self.asset.joint_names.index(j))
        self.all_joint_idx_dataset, self.all_joint_idx_asset = all_j_m, all_j_a
        self.all_joint_idx_dataset = torch.tensor(self.all_joint_idx_dataset, device=self.device)
        self.all_joint_idx_asset = torch.tensor(self.all_joint_idx_asset, device=self.device)

        self.last_reset_env_ids = None

        self._cum_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._cum_root_pos_scale = cum_root_pos_scale
        self._cum_keypoint_scale = cum_keypoint_scale
        self._cum_orientation_scale = cum_orientation_scale

        self.body_z_terminate_thres = body_z_terminate_thres
        self.body_z_terminate_patterns = body_z_terminate_patterns
        self.reinit_prob = reinit_prob
        self.reinit_min_steps = reinit_min_steps
        self.reinit_max_steps = reinit_max_steps
        self.gravity_terminate_thres = gravity_terminate_thres
        self.body_z_idx_motion, self.body_z_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.body_z_terminate_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )

        self.feet_standing = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        self.lengths = torch.full((self.num_envs,), 1, dtype=torch.int32, device=self.device)
        self.t = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.boot_indicator = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.boot_indicator_max = boot_indicator_max

        self.joint_pos_boot_protect = self.asset.data.default_joint_pos.clone()
        self.next_init_t = torch.full((self.num_envs,), -1, dtype=torch.int32, device=self.device)
        self._reinit_requested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        ## init noise
        self.init_noise_params = init_noise
        ## reward sigma
        self.reward_sigma = reward_sigma

    def sample_init(self, env_ids: torch.Tensor):
        t = self.t[env_ids]
        lengths = self.lengths[env_ids]
        self.last_reset_env_ids = env_ids
        # resample motion
        lengths = self.dataset.reset(env_ids)

        n = env_ids.shape[0]
        rand_vals = torch.rand(n, 2, device=self.device)
        
        cached_t = torch.full_like(t, -1)
        if (
            (not self.debug_mode)
            and self.reinit_prob > 0.0
            and self.reinit_max_steps > 0
            and self.reinit_max_steps >= self.reinit_min_steps
        ):
            chosen_mask = self._reinit_requested[env_ids] & (rand_vals[:, 0] < self.reinit_prob)
            rewind = (rand_vals[:, 1] * (self.reinit_max_steps - self.reinit_min_steps + 1)).to(self.t.dtype) + self.reinit_min_steps
            cached_t = torch.where(chosen_mask, (t - rewind).clamp_min(0), cached_t)
            self._reinit_requested[env_ids] = False

        if self.debug_mode:
            t[:] = 0
        else:
            max_start = lengths - self.future_steps[-1] - 1
            offsets = (rand_vals[:, 0] * 0.75 * max_start.to(torch.float32)).floor().to(self.t.dtype)
            t_rand = offsets * (rand_vals[:, 1] > self.zero_init_prob)
            t[:] = torch.where(cached_t >= 0, cached_t.clamp(max=max_start), t_rand)

        self.lengths[env_ids] = lengths
        self.t[env_ids] = t

        motion = self.dataset.get_slice(env_ids, self.t[env_ids], 1)

        # set robot state
        self.sample_init_robot(env_ids, motion)
        self.next_init_t[env_ids] = -1
        return None

    def sample_init_robot(self, env_ids: Sequence[int], motion, lift_height: float = 0.04):
        # Get subsets for the current envs
        init_root_state = self.init_root_state[env_ids].clone()
        init_joint_pos = self.init_joint_pos[env_ids].clone()
        init_joint_vel = self.init_joint_vel[env_ids].clone()
        env_origins = self.env.scene.env_origins[env_ids]
        num_envs = len(env_ids)

        # Extract motion data
        motion_root_pos = motion.root_pos_w[:, 0]
        motion_root_quat = motion.root_quat_w[:, 0]
        motion_root_lin_vel = motion.root_lin_vel_w[:, 0]
        motion_root_ang_vel = motion.root_ang_vel_w[:, 0]
        motion_joint_pos = motion.joint_pos[:, 0]
        motion_joint_vel = motion.joint_vel[:, 0]

        # -------- root state ----------------------------------------------------
        init_root_state[:, :3] = env_origins + motion_root_pos
        init_root_state[:, 2] += lift_height
        root_pos_noise = torch.randn_like(init_root_state[:, :3]).clamp(-1, 1) * self.init_noise_params["root_pos"]
        root_pos_noise[:, 2].clamp_min_(0.0)
        init_root_state[:, :3] += root_pos_noise

        init_root_state[:, 3:7] = motion_root_quat
        random_axis = torch.rand(num_envs, 3, device=self.device)
        random_angle = torch.randn(num_envs, device=self.device).clamp(-1, 1) * self.init_noise_params["root_ori"]
        random_quat = quat_from_angle_axis(random_angle, random_axis)
        init_root_state[:, 3:7] = quat_mul(random_quat, init_root_state[:, 3:7])

        init_root_state[:, 7:10] = motion_root_lin_vel
        lin_vel_noise = torch.randn_like(init_root_state[:, 7:10]).clamp(-1, 1) * self.init_noise_params["root_lin_vel"]
        init_root_state[:, 7:10] += lin_vel_noise
        
        init_root_state[:, 10:13] = motion_root_ang_vel
        ang_vel_noise = torch.randn_like(init_root_state[:, 10:13]).clamp(-1, 1) * self.init_noise_params["root_ang_vel"]
        init_root_state[:, 10:13] += ang_vel_noise

        # -------- joint state ----------------------------------------------------
        init_joint_pos[:, self.all_joint_idx_asset] = motion_joint_pos[:, self.all_joint_idx_dataset]
        init_joint_vel[:, self.all_joint_idx_asset] = motion_joint_vel[:, self.all_joint_idx_dataset]
        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * self.init_noise_params["joint_pos"]
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * self.init_noise_params["joint_vel"]
        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        # Apply the calculated states to the simulation
        self.asset.write_root_state_to_sim(init_root_state, env_ids=env_ids)

        self.joint_pos_boot_protect[env_ids] = init_joint_pos

        self.asset.write_joint_position_to_sim(init_joint_pos, env_ids=env_ids)
        self.asset.set_joint_position_target(init_joint_pos, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(init_joint_vel, env_ids=env_ids)

        self.asset.write_data_to_sim()
    
    def reset(self, env_ids):
        self.finished[env_ids] = False
        self.boot_indicator[env_ids] = self.boot_indicator_max
        self._cum_error[env_ids] = 0.0

    @termination
    def body_z_termination(self):
        if self.body_z_terminate_thres <= 0 or self.body_z_idx_asset.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        target_z = self._motion.body_pos_w[:, 0, self.body_z_idx_motion, 2]
        current_z = self.asset.data.body_pos_w[:, self.body_z_idx_asset, 2]
        exceed = (target_z.sub(current_z).abs() > self.body_z_terminate_thres).any(dim=1, keepdim=True)
        self._reinit_requested.logical_or_(exceed.view(-1))
        return exceed

    @termination
    def gravity_dir_termination(self):
        if self.gravity_terminate_thres <= 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        motion_quat = self._motion.root_quat_w[:, 0]
        motion_g_b = quat_apply_inverse(motion_quat, self.asset.data.GRAVITY_VEC_W)
        current_quat = self.asset.data.root_quat_w
        robot_g_b = quat_apply_inverse(current_quat, self.asset.data.GRAVITY_VEC_W)
        exceed = (motion_g_b[:, 2:] - robot_g_b[:, 2:]).abs() > self.gravity_terminate_thres
        self._reinit_requested.logical_or_(exceed.view(-1))
        return exceed

    @observation
    def command(self):
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        root_quat_future = self._motion.root_quat_w[:, 0:, :]
        root_quat_future0 = self._motion.root_quat_w[:, 0, :].unsqueeze(1)

        root_pos_future = self._motion.root_pos_w[:, 1:, :]
        root_pos_future0 = self._motion.root_pos_w[:, 0, :].unsqueeze(1)

        # pos diff is applied in expected root frame
        pos_diff_b = quat_apply_inverse(
            root_quat_future0,
            root_pos_future - root_pos_future0
        )
        
        # quat diff is applied in current root frame
        # because we can get reliable quat from real robot IMU
        root_quat = root_quat.expand(-1, root_quat_future.shape[1], -1)
        quat_diff = quat_mul(quat_conjugate(root_quat), root_quat_future)
        rotmat_diff = matrix_from_quat(quat_diff)
        rot6d_diff = rotmat_diff[..., :, :2].transpose(-2, -1)

        return torch.cat([
            pos_diff_b.reshape(self.num_envs, -1),
            rot6d_diff.reshape(self.num_envs, -1),
        ], dim=-1)

    def command_sym(self):
        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]).repeat(len(self.future_steps) - 1),
            sym_utils.SymmetryTransform(
                perm=torch.arange(6),
                signs=[1, -1, 1, -1, 1, -1]
            ).repeat(len(self.future_steps)),
        ])

    @observation
    def target_root_z_obs(self):
        return self._motion.root_pos_w[:, :, 2].reshape(self.num_envs, -1)
    def target_root_z_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]).repeat(len(self.future_steps))


    @observation
    def target_pos_b_obs(self):
        current_pos = self.asset.data.root_pos_w.unsqueeze(1) - self.env.scene.env_origins.unsqueeze(1)
        current_quat = self.asset.data.root_quat_w.unsqueeze(1)
        target_pos_b = quat_apply_inverse(
            current_quat,
            (self._motion.root_pos_w - current_pos)
        )
        return target_pos_b.reshape(self.num_envs, -1)
    def target_pos_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))
    
    @observation
    def target_linvel_b_obs(self):
        target_linvel_b = quat_apply_inverse(self.asset.data.root_quat_w.unsqueeze(1), self._motion.root_lin_vel_w)
        return target_linvel_b.reshape(self.num_envs, -1)
    def target_linvel_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_projected_gravity_b(self):
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        g_b = quat_apply_inverse(self._motion.root_quat_w, gravity)  # [N, S, 3]
        return g_b.reshape(self.num_envs, -1)

    def target_projected_gravity_b_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_keypoints_b_obs(self):
        target_keypoints_b = self._motion.body_pos_b[:, :, self.keypoint_idx_motion]
        return target_keypoints_b.reshape(self.num_envs, -1)
    def target_keypoints_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1]).repeat(len(self.future_steps))
    
    @observation
    def target_keypoints_diff_b_obs(self):
        actual_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset] - self.env.scene.env_origins.unsqueeze(1)
        target_w = self._motion.body_pos_w[:, :, self.keypoint_idx_motion]
        diff_w = target_w - actual_w.unsqueeze(1)
        diff_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1).unsqueeze(1),
            diff_w
        )
        return diff_b.reshape(self.num_envs, -1)
    def target_keypoints_diff_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1]).repeat(len(self.future_steps))

    @observation
    def relative_quat_obs(self):
        relative_quat = quat_mul(
            quat_conjugate(self.asset.data.root_quat_w.unsqueeze(1)),
            self._motion.root_quat_w
        )
        rotmat = matrix_from_quat(relative_quat)
        rot6d = rotmat[..., :, :2].transpose(-2, -1)
        return rot6d.reshape(self.num_envs, -1)
    def relative_quat_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(6),
            signs=[1, -1, 1, -1, 1, -1]
        ).repeat(len(self.future_steps))

    @observation
    def target_joint_pos_obs(self):
        return self._motion.joint_pos.reshape(self.num_envs, -1)
    def target_joint_pos_obs_sym(self):
        return sym_utils.joint_space_symmetry(self.asset, self.dataset.joint_names).repeat(len(self.future_steps))


    @observation
    def current_keypoint_b(self):
        actual_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        actual_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            actual_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        return actual_b.reshape(self.num_envs, -1)
    def current_keypoint_b_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])

    @observation
    def current_keypoint_vel_b(self):
        actual_vel_w = self.asset.data.body_lin_vel_w[:, self.keypoint_idx_asset]
        actual_vel_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            actual_vel_w
        )
        return actual_vel_b.reshape(self.num_envs, -1)
    def current_keypoint_vel_b_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])
    
    @observation
    def boot_indicator_state(self):
        return self.boot_indicator / self.boot_indicator_max
    def boot_indicator_state_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1.])

    @reward
    def root_pos_tracking(self):
        current_pos = self.asset.data.root_pos_w
        target_pos = self.reward_root_pos_w
        diff = target_pos - current_pos
        error = diff.norm(dim=-1, keepdim=True)
        self._cum_error[:, 0:1] = error / self._cum_root_pos_scale
        return _calc_exp_sigma(error, self.reward_sigma["root_pos"])

    @reward
    def root_vel_tracking(self):
        current_linvel_w = self.asset.data.root_lin_vel_w
        current_quat = self.asset.data.root_quat_w
        ref_linvel_w = self._motion.root_lin_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_linvel_b = quat_apply_inverse(current_quat, current_linvel_w)
        ref_linvel_b = quat_apply_inverse(ref_quat, ref_linvel_w)
        diff = ref_linvel_b - current_linvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["root_vel"])

    @reward
    def root_rot_tracking(self):
        current_quat = self.asset.data.root_quat_w
        target_quat = self.reward_root_quat_w
        diff = axis_angle_from_quat(quat_mul(
            target_quat,
            quat_conjugate(current_quat)
        ))
        error = torch.norm(diff, dim=-1, keepdim=True)
        self._cum_error[:, 1:2] = error / self._cum_orientation_scale
        return _calc_exp_sigma(error, self.reward_sigma["root_rot"])
    
    @reward
    def root_ang_vel_tracking(self):
        current_angvel_w = self.asset.data.root_ang_vel_w
        current_quat = self.asset.data.root_quat_w
        ref_angvel_w = self._motion.root_ang_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_angvel_b = quat_apply_inverse(current_quat, current_angvel_w)
        ref_angvel_b = quat_apply_inverse(ref_quat, ref_angvel_w)
        diff = ref_angvel_b - current_angvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["root_ang_vel"])

    @reward
    def keypoint_tracking(self):
        return self._keypoint_tracking(
            self.keypoint_idx_asset,
            self.keypoint_idx_motion,
            "keypoint",
            update_cum_error=True,
        )
    
    @reward
    def lower_keypoint_tracking(self):
        return self._keypoint_tracking(
            self.lower_keypoint_idx_asset,
            self.lower_keypoint_idx_motion,
            "lower_keypoint",
        )

    @reward
    def upper_keypoint_tracking(self):
        return self._keypoint_tracking(
            self.upper_keypoint_idx_asset,
            self.upper_keypoint_idx_motion,
            "upper_keypoint",
        )

    @reward
    def keypoint_vel_tracking(self):
        current_root_quat = self.asset.data.root_quat_w
        actual_vel_w = self.asset.data.body_lin_vel_w[:, self.keypoint_idx_asset]
        actual_vel_b = quat_apply_inverse(
            current_root_quat.unsqueeze(1),
            actual_vel_w - self.asset.data.root_lin_vel_w.unsqueeze(1),
        )

        target_vel_b = self._motion.body_vel_b[:, 0, self.keypoint_idx_motion]
        error = (target_vel_b - actual_vel_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["keypoint_vel"])

    @reward
    def keypoint_rot_tracking(self):
        current_root_quat = self.asset.data.root_quat_w
        actual_quat_b = quat_mul(
            quat_conjugate(current_root_quat).unsqueeze(1),
            self.asset.data.body_quat_w[:, self.keypoint_idx_asset],
        )
        target_quat_b = self._motion.body_quat_b[:, 0, self.keypoint_idx_motion]
        diff = axis_angle_from_quat(quat_mul(target_quat_b, quat_conjugate(actual_quat_b)))
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["keypoint_rot"])

    @reward
    def keypoint_angvel_tracking(self):
        current_root_quat = self.asset.data.root_quat_w
        actual_angvel_w = self.asset.data.body_ang_vel_w[:, self.keypoint_idx_asset]
        root_angvel_w = self.asset.data.root_ang_vel_w.unsqueeze(1)
        actual_angvel_b = quat_apply_inverse(
            current_root_quat.unsqueeze(1),
            actual_angvel_w - root_angvel_w,
        )
        target_angvel_b = self._motion.body_angvel_b[:, 0, self.keypoint_idx_motion]
        error = (target_angvel_b - actual_angvel_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["keypoint_angvel"])

    def _keypoint_tracking(
        self,
        keypoint_idx_asset: torch.Tensor,
        keypoint_idx_motion: torch.Tensor,
        sigma_key: str,
        update_cum_error: bool = False,
    ):
        actual = self.asset.data.body_pos_w[:, keypoint_idx_asset]
        target = self.reward_keypoints_w[:, keypoint_idx_motion]
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        if update_cum_error:
            self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return _calc_exp_sigma(error, self.reward_sigma[sigma_key])

    @reward
    def joint_pos_tracking(self):
        actual = self.asset.data.joint_pos[:, self.joint_idx_asset]
        target = self._motion.joint_pos[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["joint_pos"])

    @reward
    def joint_vel_tracking(self):
        actual = self.asset.data.joint_vel[:, self.joint_idx_asset]
        target = self._motion.joint_vel[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["joint_vel"])

    def update_reward_target_raw(self):
        delta_quat = quat_mul(
            self.asset.data.root_quat_w,
            quat_conjugate(self._motion.root_quat_w[:, 0])
        )
        tgt_rel = self._motion.body_pos_w[:, 0] - self._motion.root_pos_w[:, 0].unsqueeze(1)
        self.reward_keypoints_w = quat_apply(delta_quat.unsqueeze(1), tgt_rel) + self.asset.data.root_pos_w.unsqueeze(1)

        if not self.student_train:
            self.reward_root_pos_w = self._motion.root_pos_w[:, 0] + self.env.scene.env_origins
            self.reward_root_quat_w = self._motion.root_quat_w[:, 0]
        else:
            steps = 50  # calc t+50 target root pos/rot from current root pos/rot
            # prepare future root pos/rot cache
            if hasattr(self, 'ts_root_pos_w') is False:
                self.ts_root_pos_w = torch.zeros(self.num_envs, steps, 3, device=self.device, dtype=torch.float32)
            # update only for reset envs
            if self.last_reset_env_ids is not None:
                future_motion = self.dataset.get_slice(self.last_reset_env_ids, self.t[self.last_reset_env_ids], steps=steps)
                self.ts_root_pos_w[self.last_reset_env_ids] = future_motion.root_pos_w + self.env.scene.env_origins[self.last_reset_env_ids].unsqueeze(1)
            # get current root pos/rot from cache
            reward_pos = self.ts_root_pos_w[:, 0].clone()
            # roll forward the cache
            self.ts_root_pos_w[:, :-1] = self.ts_root_pos_w[:, 1:]
            # compute target root pos/rot at t+steps
            current_pos_t = self.asset.data.root_pos_w
            current_quat_t = self.asset.data.root_quat_w

            ref_motion_plus = self.dataset.get_slice(None, self.t, steps=torch.tensor([steps], device=self.device, dtype=torch.int64))
            ref_pos_t = self._motion.root_pos_w[:, 0]
            ref_pos_t_plus = ref_motion_plus.root_pos_w[:, 0]
            ref_quat_t = self._motion.root_quat_w[:, 0]

            delta_quat = quat_mul(current_quat_t, quat_conjugate(ref_quat_t))
            self.ts_root_pos_w[:, -1] = quat_apply(delta_quat, (ref_pos_t_plus - ref_pos_t)) + current_pos_t

            self.reward_root_pos_w = reward_pos
            self.reward_root_quat_w = ref_quat_t

    def before_update(self):
        self.t = torch.clamp_max(self.t + 1, self.lengths - 1)
        self.finished[:] = self.t >= self.lengths - 1
        self.boot_indicator[:] = torch.clamp_min(self.boot_indicator - 1, 0)

        self._motion = self.dataset.get_slice(None, self.t, steps=self.future_steps)

        self.feet_standing = (self._motion.body_vel_w[:, 0, self.feet_idx_motion, :].norm(dim=-1, keepdim=False) < 0.2) & (self._motion.body_pos_w[:, 0, self.feet_idx_motion, 2] < 0.15)

        self.update_reward_target_raw()

    def update(self):
        self.dataset.update()
        if self.last_reset_env_ids is not None:
            self.last_reset_env_ids = None

    def debug_draw(self):
        root_pos = self.asset.data.root_pos_w    # [N,1,3]
        root_quat = self.asset.data.root_quat_w  # [N,1,4]
        target_root_quat = self.reward_root_quat_w  # [N,1,4]
        heading_rel = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        heading_world = quat_apply(root_quat, heading_rel)
        heading_world_target = quat_apply(target_root_quat, heading_rel)

        # —— original world‐frame drawing —— 
        target_keypoints_w = self.reward_keypoints_w[:, self.keypoint_idx_motion]
        robot_keypoints_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]

        # draw points and error vectors
        self.env.debug_draw.point(
            target_keypoints_w.reshape(-1, 3), color=(1, 0, 0, 1)
        )
        self.env.debug_draw.point(
            robot_keypoints_w.reshape(-1, 3), color=(0, 1, 0, 1)
        )
        self.env.debug_draw.vector(
            robot_keypoints_w.reshape(-1, 3),
            (target_keypoints_w - robot_keypoints_w).reshape(-1, 3),
            color=(0, 0, 1, 1)
        )
        
        self.env.debug_draw.vector(
            root_pos.reshape(-1, 3),
            heading_world.reshape(-1, 3),
            color=(0, 0, 1, 2)
        )
        
        self.env.debug_draw.vector(
            self.reward_root_pos_w.reshape(-1, 3),
            heading_world_target.reshape(-1, 3),
            color=(1, 0, 0, 2)
        )
        if self.feet_idx_motion.numel() > 0 and self.feet_standing.any():
            target_feet_w = self.reward_keypoints_w[:, self.feet_idx_motion]
            standing_points = target_feet_w[self.feet_standing]
            if standing_points.numel() > 0:
                self.env.debug_draw.point(
                    standing_points,
                    color=(1, 1, 0, 1),
                    size=20.0,
                )
