import torch
import einops
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
import isaaclab.utils.string as string_utils
import active_adaptation.utils.symmetry as symmetry_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
    
    def symmetry_transforms(self):
        raise NotImplementedError(
            "ActionManager subclasses must implement symmetry_transforms method."
            "This method should return a SymmetryTransform object that applies to the action space."
        )


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int | None = None,
        alpha: Tuple[float, float] = (0.9, 0.9),
        alpha_wide: Tuple[float, float] = (0.8, 1.0),
        boot_protect: bool = False,
        alpha_jit_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(env)

        # ------------------------------------------------------------------ cfg
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )
        # print(self.joint_ids, self.joint_names, self.action_scaling)
        # breakpoint()
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        self.max_delay = max_delay or 0  # physics steps

        self.alpha_range = alpha
        self.alpha_wide_range = alpha_wide

        # Boot‑protection ----------------------------------------------------
        self.boot_protect_enabled = boot_protect
        if self.boot_protect_enabled:
            self.boot_delay = torch.zeros(self.num_envs, 1, dtype=int, device=self.device)

        # α‑jitter -----------------------------------------------------------
        self.alpha_jit_scale = alpha_jit_scale
        if self.alpha_jit_scale is not None:
            self.alpha_jit = torch.zeros(self.num_envs, 1, device=self.device)

        # Persistent tensors -------------------------------------------------
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)

        with torch.device(self.device):
            hist = max((self.max_delay - 1) // self.env.decimation + 1, 3)
            self.action_buf = torch.zeros(self.num_envs, hist, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    # --------------------------------------------------------------------- util
    def resolve(self, spec, names=None):
        """Convenience helper for user APIs."""
        target_names = self.asset.joint_names if names is None else names
        return string_utils.resolve_matching_names_values(dict(spec), target_names)
    
    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform
    # ------------------------------------------------------------------- reset
    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        # Delay selection ---------------------------------------------------
        if self.boot_protect_enabled:
            delay = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
            self.boot_delay[env_ids] = delay
            self.delay[env_ids] = delay
        else:
            self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)

        # α per environment --------------------------------------------------
        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.alpha_range)
        self.alpha[env_ids] = alpha

    # ---------------------------------------------------------------- forward
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            raw_action = tensordict["action"].clamp(-10, 10)

            ### debug symmetry
            # raw_action = self.symmetry_transforms().to(raw_action.device).forward(raw_action)

            # α with optional jitter -------------------------------------------
            if self.alpha_jit_scale is not None:
                self.alpha_jit.uniform_(-self.alpha_jit_scale, self.alpha_jit_scale)
                self.alpha.add_(self.alpha_jit).clamp_(*self.alpha_wide_range)

            self.action_buf = torch.roll(self.action_buf, shifts=1, dims=1)
            self.action_buf[:, 0, :] = raw_action

        # Communication delay ----------------------------------------------
        idx = (self.delay - substep + self.env.decimation - 1) // self.env.decimation
        delayed_action = self.action_buf.take_along_dim(idx.unsqueeze(1), dim=1).squeeze(1)
        self.applied_action.lerp_(delayed_action, self.alpha)

        # Joint targets -----------------------------------------------------
        pos_tgt = self.default_joint_pos + self.offset
        pos_tgt[:, self.joint_ids] += self.applied_action * self.action_scaling

        # Optional boot‑protection -----------------------------------------
        if self.boot_protect_enabled:
            pos_tgt = torch.where(
                self.boot_delay > 0,
                self.env.command_manager.joint_pos_boot_protect,
                pos_tgt,
            )
            self.boot_delay.sub_(1).clamp_min_(0)

        # Write to simulator -----------------------------------------------
        self.asset.set_joint_position_target(pos_tgt)
        self.asset.write_data_to_sim()
