import numpy as np
import simple_observation
import onnxruntime as ort
import json
import mujoco
import mujoco.viewer
import yaml
from pathlib import Path
from common.joint_mapper import (
    JointMapper,
    create_real_to_mujoco_mapper,
    create_isaac_to_real_mapper,
)
from typing import Dict, List, Optional, Tuple
from common.utils import DictToClass, Timer
from paths import ASSETS_DIR, REAL_G1_ROOT

from scipy.spatial.transform import Rotation as R
from common.math_utils import (
    _linspace_rows,
    _remove_yaw_keep_rp_wxyz,
    _slerp,
    _yaw_component_wxyz,
    _zero_z,
)


# Helper Utilities
class ONNXModule:
    def __init__(self, path: str):
        self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta_path = path.replace(".onnx", ".json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.in_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]]
        self.out_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]]

    def __call__(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        args = {
            inp.name: input[key]
            for inp, key in zip(self.ort_session.get_inputs(), self.in_keys)
            if key in input
        }
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.out_keys, outputs)}
        return outputs

def mapping_joints(data: np.ndarray, target: List[str]):
    from common.utils import joint_names_29, joint_names_23
    nums = data.shape[-1]
    if nums == len(target):
        return data
    if nums == 29:
        current = joint_names_29
        print("[Mapping] from 29 to 23")
    elif nums == 23:
        current = joint_names_23
        print("[Mapping] from 23 to 29")
    else:
        raise ValueError(f"Unsupported number of joints: {nums}")

    new_data = np.zeros((data.shape[0], len(target)), dtype=np.float32)
    for i, name in enumerate(target):
        if name in current:
            new_data[:, i] = data[:, current.index(name)]
    return new_data.astype(np.float32)


class Controller:
    def __init__(self, config, policy_cfg):
        self.config = config
        self.policy_cfg = policy_cfg
        self.action_scale_isaac = self.policy_cfg.action_scale

        ckpt_path = self.policy_cfg.policy_path
        self.module = ONNXModule(ckpt_path)

        self.isaac_to_real_mapper_state = create_isaac_to_real_mapper(
            self.config.isaac_joint_names_state,
            self.config.real_joint_names
        )

        self._init_buffers()
        self._load_motions()
        self._build_obs_modules()

    def _init_buffers(self):
        self.policy_input = None
        self.last_action = np.zeros(len(self.config.mujoco_joint_names), dtype=np.float32)
        self.action_clip = float(self.policy_cfg.action_clip)

        self.dof_size_real = len(self.config.real_joint_names)
        self.default_qpos_real = np.array(self.config.default_qpos_real, dtype=np.float32)

        self.smoothing_alpha = getattr(self.policy_cfg, "lowstate_alpha", 0.2)
        self._qj_smooth   = np.zeros(self.dof_size_real, dtype=np.float32)
        self._dqj_smooth  = np.zeros(self.dof_size_real, dtype=np.float32)
        self._tau_smooth  = np.zeros(self.dof_size_real, dtype=np.float32)
        self._quat_smooth = np.zeros(4, dtype=np.float32)
        self._gyro_smooth = np.zeros(3, dtype=np.float32)

        self.qj_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.dqj_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.tau_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)

        # ---- Reference stream ----------------------------------------------
        self.ref_joint_pos: Optional[np.ndarray] = None  # (T_ref, J)
        self.ref_root_quat: Optional[np.ndarray] = None  # (T_ref, 4)
        self.ref_root_pos: Optional[np.ndarray] = None   # (T_ref, 3)

        # ---- Playback state ------------------------------------------------
        self.ref_idx: int = 0
        self.ref_len: int = 0
        self.current_name: str = "default"
        self.current_done: bool = True  # boot: default done

        self.n_joints = len(self.policy_cfg.dataset_joint_names)

        self.transition_steps = int(getattr(self.policy_cfg, "transition_steps", 100))

    def _load_motions(self):
        self.motions: Dict[str, Dict[str, np.ndarray]] = {}
        for m in self.policy_cfg.motions:
            mc = DictToClass(m)
            motion_name = mc.name
            mp = Path(mc.path)
            path = str(mp if mp.is_absolute() else (REAL_G1_ROOT / mp))
            t0, t1 = int(mc.start), int(mc.end)

            data = np.load(path, allow_pickle=True)
            # if not isinstance(data, np.lib.npyio.NpzFile):
            #     print("[DEBUG] path repr =", repr(path))
            #     print("[DEBUG] endswith .npz =", str(path).endswith(".npz"))
            #     raise ValueError(f"[TrackingPolicyRaw] Only .npz is supported: {path}")

            joint_pos = data["dof_pos"][t0:t1].astype(np.float32)
            root_pos = data["root_pos"][t0:t1].astype(np.float32)
            root_rot_xyzw = data["root_rot"][t0:t1].astype(np.float32)
            root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)

            joint_names = data.get("joint_names", None)
            if joint_names is not None:
                if isinstance(joint_names, list):
                    pass
                else:
                    joint_names = joint_names.tolist()
                target_names = list(self.policy_cfg.dataset_joint_names)
                if joint_names != target_names:
                    name_to_idx = {n: i for i, n in enumerate(joint_names)}
                    remap = np.zeros((joint_pos.shape[0], len(target_names)), dtype=np.float32)
                    for i, n in enumerate(target_names):
                        j = name_to_idx.get(n, None)
                        if j is not None:
                            remap[:, i] = joint_pos[:, j]
                    joint_pos = remap

            self.motions[motion_name] = {
                "joint_pos": joint_pos,  # (T,J)
                "root_quat": root_quat,  # (T,4) wxyz
                "root_pos": root_pos,    # (T,3)
            }
        # ---- One-frame motion clips (config provided) ----------------------
        for m in self.policy_cfg.motion_clips:
            mc = DictToClass(m)
            motion_name = mc.name
            joint_pos_1 = mapping_joints(
                np.asarray(mc.joint_pos, dtype=np.float32).reshape(1, -1),
                self.policy_cfg.dataset_joint_names
            )
            root_quat_1 = np.asarray(mc.root_quat, dtype=np.float32).reshape(1, 4)
            root_pos_1 = np.asarray(mc.root_pos, dtype=np.float32).reshape(1, 3)

            self.motions[motion_name] = {
                "joint_pos": joint_pos_1,  # (1,J)
                "root_quat": root_quat_1,  # (1,4)
                "root_pos": root_pos_1,    # (1,3)
            }

        assert "default" in self.motions, "[TrackingPolicyRaw] motions must include a 'default' clip (length==1)."

    def _read_current_state(self) -> Dict[str, np.ndarray]:
        q_real = self.qj_real.copy()
        real_names = list(self.config.real_joint_names)
        target_names = list(self.policy_cfg.dataset_joint_names)
        name_to_idx = {n: i for i, n in enumerate(real_names)}
        q_policy = np.zeros(len(target_names), dtype=np.float32)
        for i, n in enumerate(target_names):
            j = name_to_idx.get(n, None)
            if j is not None and j < q_real.shape[0]:
                q_policy[i] = q_real[j]

        if self.ref_root_pos is not None:
            root_pos = self.ref_root_pos[self.ref_idx]
            root_quat = self.ref_root_quat[self.ref_idx]
        else:
            root_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32)
            root_quat = self.quat.copy()
        print(root_pos)
        return {
            "joint_pos": q_policy.astype(np.float32),
            "root_pos": root_pos,
            "root_quat": root_quat,
        }

    def _align_motion_to_current(
        self,
        motion: Dict[str, np.ndarray],
        curr: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        p0 = motion["root_pos"][0]
        q0_yaw = _yaw_component_wxyz(motion["root_quat"][0])
        pc = curr["root_pos"]
        qc_yaw = _yaw_component_wxyz(curr["root_quat"])

        R0 = R.from_quat(q0_yaw, scalar_first=True)
        Rc = R.from_quat(qc_yaw, scalar_first=True)
        R_delta = Rc * R0.inv()

        root_pos_aligned = R_delta.apply(motion["root_pos"] - p0) + pc
        root_pos_aligned[:, 2] = motion["root_pos"][:, 2]  # keep original z

        root_quat_all = R.from_quat(motion["root_quat"], scalar_first=True)
        root_quat_aligned = (R_delta * root_quat_all).as_quat(scalar_first=True)

        return {
            "joint_pos": motion["joint_pos"].astype(np.float32).copy(),
            "root_quat": root_quat_aligned.astype(np.float32),
            "root_pos": root_pos_aligned.astype(np.float32),
        }

    def _build_transition_prefix(
        self,
        curr: Dict[str, np.ndarray],
        tgt_first: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        T = int(self.transition_steps)
        if T <= 0:
            raise ValueError("[TrackingPolicyRaw] transition_steps must be > 0")

        joints_tr = _linspace_rows(curr["joint_pos"], tgt_first["joint_pos"], T)
        root_pos_tr = _linspace_rows(curr["root_pos"], tgt_first["root_pos"], T)
        root_quat_tr = _slerp(curr["root_quat"], tgt_first["root_quat"], T)

        return {
            "joint_pos": joints_tr,
            "root_quat": root_quat_tr,
            "root_pos": root_pos_tr,
        }

    def _start_motion_from_current(self, name: str):
        assert name in self.motions
        curr = self._read_current_state()

        m = self.motions[name]
        aligned_motion = self._align_motion_to_current(m, curr)

        tgt_first = {
            "joint_pos": aligned_motion["joint_pos"][0],
            "root_quat": aligned_motion["root_quat"][0],
            "root_pos": aligned_motion["root_pos"][0],
        }

        trans_motion = self._build_transition_prefix(curr, tgt_first)

        self.ref_joint_pos = np.concatenate([trans_motion["joint_pos"], aligned_motion["joint_pos"]], axis=0)
        self.ref_root_quat = np.concatenate([trans_motion["root_quat"], aligned_motion["root_quat"]], axis=0)
        self.ref_root_pos = np.concatenate([trans_motion["root_pos"], aligned_motion["root_pos"]], axis=0)

        self.ref_idx = 0
        self.ref_len = int(self.ref_joint_pos.shape[0])
        self.current_name = name
        self.current_done = (self.ref_len <= 1)

        print(f"[TrackingPolicyRaw] Start motion '{name}' | ref_len={self.ref_len}, transition={self.transition_steps}")

    def _build_obs_modules(self):
        from simple_observation import (
            TrackingCommandObsRaw,
            TargetRootZObs,
            TargetJointPosObs,
            TargetProjectedGravityBObs,
            RootAngVelB,
            ProjectedGravityB,
            JointPos,
            PrevActions,
            BootIndicator,
        )
        self.obs_modules = [
            BootIndicator(),
            TrackingCommandObsRaw(self, self),
            TargetRootZObs(self),
            TargetJointPosObs(self),
            TargetProjectedGravityBObs(self),
            RootAngVelB(self),
            ProjectedGravityB(self),
            JointPos(self, pos_steps=[0, 1, 2, 3, 4, 8]),
            PrevActions(self, steps=3),
        ]
        self.num_obs = sum(m.size for m in self.obs_modules)

    def _update_obs(self):
        if self.ref_len > 0 and self.ref_idx < self.ref_len - 1:
            self.ref_idx += 1
            if self.ref_idx == self.ref_len - 1:
                self.current_done = True

        obs_list = []
        for m in self.obs_modules:
            m.update()
            obs_list.append(m.compute())
        if self.policy_input is None:
            self.policy_input = {
                "policy": np.zeros((1, self.num_obs), dtype=np.float32),
                "is_init": np.ones((1,), dtype=bool),
            }
        else:
            self.policy_input["policy"][0, :] = np.concatenate(obs_list, axis=0)

    def process_states(self, state_msg):
        a = self.smoothing_alpha

        q = state_msg["q"]
        dq = state_msg["dq"]
        tau = state_msg["tau"]

        imu_state_quaternion = state_msg["imu_state_quaternion"]
        imu_state_gyroscope = state_msg["imu_state_gyroscope"]

        # NOTE!!! Originally, this filter is applied at Frequency=200Hz, but now only 50Hz
        # NOTE!!! Cause smoothing_alpha=1, thus currently we fully trust q from sensors
        # NOTE!!! So frequency doesn't affect behavior for now
        self._qj_smooth[:]   = (1 - a) * self._qj_smooth[:]  + a * np.array(q, dtype=np.float32)
        self._dqj_smooth[:]  = (1 - a) * self._dqj_smooth[:] + a * np.array(dq, dtype=np.float32)
        self._tau_smooth[:]  = (1 - a) * self._tau_smooth[:] + a * np.array(tau, dtype=np.float32)
        self._quat_smooth[:] = (1 - a) * self._quat_smooth   + a * np.array(imu_state_quaternion, dtype=np.float32)
        self._gyro_smooth[:] = (1 - a) * self._gyro_smooth    + a * np.array(imu_state_gyroscope, dtype=np.float32)

        self.qj_real[:]  = self._qj_smooth
        self.dqj_real[:] = self._dqj_smooth
        self.tau_real[:] = self._tau_smooth
        self.quat[:]     = self._quat_smooth
        self.gyro[:]     = self._gyro_smooth

        self.qj_isaac = self.isaac_to_real_mapper_state.map_state_to_from(self.qj_real)
        self.dqj_isaac = self.isaac_to_real_mapper_state.map_state_to_from(self.dqj_real)
        self.tau_isaac = self.isaac_to_real_mapper_state.map_state_to_from(self.tau_real)

    def inference(self, state_msg):
        """
        This module runs at 50Hz
        """
        self.process_states(state_msg)
        self._update_obs()
        out = self.module(self.policy_input)

        action_isaac = out["action"].copy()[0].astype(np.float32).clip(-self.action_clip, self.action_clip)
        self.last_action[:] = action_isaac
        applied_action_isaac = action_isaac * self.action_scale_isaac
        action_real = self.isaac_to_real_mapper_state.map_action_from_to(applied_action_isaac)
        self.policy_input["is_init"][:] = False
        desired = action_real + self.default_qpos_real
        return desired
        pass

class Sim2Sim:
    def __init__(self, config):
        model_path = "assets/g1/g1.xml"

        self.config = config
        self.default_qpos_real = np.array(self.config.default_qpos_real, dtype=np.float32)

        self.low_level_freq = 500
        self.low_level_dt = 1. / self.low_level_freq

        self.control_freq = self.config.control_freq
        self.control_dt = 1. / self.config.control_freq

        self.decimation = self.low_level_freq / self.control_freq

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.low_level_dt
        self.data = mujoco.MjData(self.model)

        self.ctrl_lower = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_upper = self.model.actuator_ctrlrange[:, 1]

        self.real_to_mujoco_mapper = create_real_to_mujoco_mapper(
            self.config.real_joint_names,
            self.config.mujoco_joint_names
        )

        # Set initial position
        self.data.qpos[7:] = self.real_to_mujoco_mapper.map_state_to_from(self.config.default_qpos_real)
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)

        # MuJoCo viewer
        self.viewer = None
        self.renderer = None
        self._viewer_tick = 0
        self.viewer_decim = max(1, self.low_level_freq // 30)  # Default 30 fps for viewer

        self.__kp_real = self.config.kps_real
        self.__kd_real = self.config.kds_real
        self.__ptargets_real = np.zeros(len(self.config.real_joint_names))

    def _viewer_sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            self.is_alive = False
            return False
        self._viewer_tick += 1
        if (self._viewer_tick % self.viewer_decim) == 0:
            self.viewer.sync()
        return True

    def _collect_state_variables(self):
        joint_qpos_mujoco = self.data.qpos[7:]  # Skip base pose
        joint_qvel_mujoco = self.data.qvel[6:]  # Skip base velocity
        joint_torque_mujoco = self.data.ctrl.copy()

        joint_qpos_real   = self.real_to_mujoco_mapper.map_state_to_from(joint_qpos_mujoco)
        joint_qvel_real   = self.real_to_mujoco_mapper.map_state_to_from(joint_qvel_mujoco)
        joint_torque_real = self.real_to_mujoco_mapper.map_state_to_from(joint_torque_mujoco)

        q = joint_qpos_real
        dq = joint_qvel_real
        tau = joint_torque_real

        imu_state_quaternion = self.data.qpos[3:7].copy() # Mujoco is wxyz
        imu_state_gyroscope = self.data.qvel[3:6].copy()

        return {
            "q": q,
            "dq": dq,
            "tau": tau,
            "imu_state_quaternion": imu_state_quaternion,
            "imu_state_gyroscope": imu_state_gyroscope,
        }

    def load_controller(self, controller: Controller):
        self.controller = controller
        state_msg = self._collect_state_variables()
        self.controller.process_states(state_msg)
        self.controller._start_motion_from_current("default")
        pass

    def run(self, cbk=None):
        cnt = 0
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            self.viewer = viewer
            timer = Timer(self.low_level_dt)
            while viewer.is_running():
                if cnt % self.decimation == 0:
                    state_msg = self._collect_state_variables()
                    desired = self.controller.inference(state_msg)
                    self.__ptargets_real = desired
                    cnt = 0

                cnt += 1
                if cbk is not None:
                    cbk()

                ptargets_mujoco = self.real_to_mujoco_mapper.map_action_from_to(self.__ptargets_real)
                kp_mujoco = self.real_to_mujoco_mapper.map_action_from_to(self.__kp_real)
                kd_mujoco = self.real_to_mujoco_mapper.map_action_from_to(self.__kd_real)

                qpos_mujoco = self.data.qpos[7:]
                qvel_mujoco = self.data.qvel[6:]
                ctrl = kp_mujoco * (ptargets_mujoco - qpos_mujoco) + kd_mujoco * (0 - qvel_mujoco)
                ctrl = np.clip(ctrl, self.ctrl_lower, self.ctrl_upper)
                self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)
                
                self._viewer_sync()
                timer.sleep()
                pass


if "__main__" == __name__:
    config_path = "config/controller.yaml"
    cfg = DictToClass(yaml.load(open(str(config_path), 'r'), Loader=yaml.FullLoader))

    policy_config_path = "config/tracking.yaml"
    policy_cfg = DictToClass(yaml.load(open(str(policy_config_path), 'r'), Loader=yaml.FullLoader))
    controller = Controller(config=cfg, policy_cfg=policy_cfg)
    sim2sim = Sim2Sim(cfg)
    sim2sim.load_controller(controller)

    def foo1():
        def foo2():
            if foo2.cnt == 2500:
                print("Started ref motion")
                sim2sim.controller._start_motion_from_current("dance1_subject1")
            foo2.cnt += 1
            pass
        foo2.cnt = 0
        return foo2
    sim2sim.run(cbk=foo1())
