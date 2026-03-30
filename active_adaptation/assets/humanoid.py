import os
import copy
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
import active_adaptation.utils.symmetry as symmetry_utils

from isaaclab.assets import ArticulationCfg as _ArticulationCfg
from isaaclab.utils import configclass

from typing import Mapping

@configclass
class ArticulationCfg(_ArticulationCfg):
    joint_symmetry_mapping: Mapping[str, list[int | tuple[int, str]]] = None
    spatial_symmetry_mapping: Mapping[str, str] = None

ASSET_PATH = os.path.dirname(__file__)

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/G1/g1_29dof_rev_1_0_flat.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.28,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_pitch_joint": 0.87,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            ".*wrist_roll_joint": 0.0,
            ".*wrist_pitch_joint": 0.0,
            ".*wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                # legs
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                # waist
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                # feet
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
                # arms
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                # legs
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                # waist
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
                # feet
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
                # arms
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                # legs (7520_14 / 7520_22)
                ".*_hip_yaw_joint": 40.17923847137318,
                ".*_hip_pitch_joint": 40.17923847137318,
                ".*_hip_roll_joint": 99.09842777666113,
                ".*_knee_joint": 99.09842777666113,
                # waist
                "waist_yaw_joint": 40.17923847137318,         # 7520_14
                "waist_roll_joint": 28.50124619574858,        # 2 * 5020
                "waist_pitch_joint": 28.50124619574858,       # 2 * 5020
                # feet
                ".*_ankle_pitch_joint": 28.50124619574858,    # 2 * 5020
                ".*_ankle_roll_joint": 28.50124619574858,     # 2 * 5020
                # arms
                ".*_shoulder_pitch_joint": 14.25062309787429, # 5020
                ".*_shoulder_roll_joint": 14.25062309787429,  # 5020
                ".*_shoulder_yaw_joint": 14.25062309787429,   # 5020
                ".*_elbow_joint": 14.25062309787429,          # 5020
                ".*_wrist_roll_joint": 14.25062309787429,     # 5020
                ".*_wrist_pitch_joint": 16.77832748089279,    # 4010
                ".*_wrist_yaw_joint": 16.77832748089279,      # 4010
            },
            damping={
                # legs (7520_14 / 7520_22)
                ".*_hip_yaw_joint": 2.5578897650279457,
                ".*_hip_pitch_joint": 2.5578897650279457,
                ".*_hip_roll_joint": 6.3088018534966395,
                ".*_knee_joint": 6.3088018534966395,
                # waist
                "waist_yaw_joint": 2.5578897650279457,        # 7520_14
                "waist_roll_joint": 1.814445686584846,        # 2 * 5020
                "waist_pitch_joint": 1.814445686584846,       # 2 * 5020
                # feet
                ".*_ankle_pitch_joint": 1.814445686584846,    # 2 * 5020
                ".*_ankle_roll_joint": 1.814445686584846,     # 2 * 5020
                # arms
                ".*_shoulder_pitch_joint": 0.907222843292423, # 5020
                ".*_shoulder_roll_joint": 0.907222843292423,  # 5020
                ".*_shoulder_yaw_joint": 0.907222843292423,   # 5020
                ".*_elbow_joint": 0.907222843292423,          # 5020
                ".*_wrist_roll_joint": 0.907222843292423,     # 5020
                ".*_wrist_pitch_joint": 1.06814150219,        # 4010
                ".*_wrist_yaw_joint": 1.06814150219,          # 4010
            },
            armature={
                # legs
                ".*_hip_yaw_joint": 0.010177520,              # 7520_14
                ".*_hip_pitch_joint": 0.010177520,            # 7520_14
                ".*_hip_roll_joint": 0.025101925,             # 7520_22
                ".*_knee_joint": 0.025101925,                 # 7520_22
                # waist
                "waist_yaw_joint": 0.010177520,               # 7520_14
                "waist_roll_joint": 0.00721945,               # 2 * 5020
                "waist_pitch_joint": 0.00721945,              # 2 * 5020
                # feet
                ".*_ankle_pitch_joint": 0.00721945,           # 2 * 5020
                ".*_ankle_roll_joint": 0.00721945,            # 2 * 5020
                # arms
                ".*_shoulder_pitch_joint": 0.003609725,       # 5020
                ".*_shoulder_roll_joint": 0.003609725,        # 5020
                ".*_shoulder_yaw_joint": 0.003609725,         # 5020
                ".*_elbow_joint": 0.003609725,                # 5020
                ".*_wrist_roll_joint": 0.003609725,           # 5020
                ".*_wrist_pitch_joint": 0.00425,              # 4010
                ".*_wrist_yaw_joint": 0.00425,                # 4010
            }
        ),
    },
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
        "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
        "right_hand_mimic": "left_hand_mimic",
        "head_mimic": "head_mimic",
    })
)

G1_COL_FULL = copy.deepcopy(G1_CFG)
G1_COL_FULL.spawn.usd_path = f"{ASSET_PATH}/G1/g1_flat_fullcol.usd"
G1_COL_FULL.spawn.articulation_props.enabled_self_collisions = False

G1_COL_FULL_SELF = copy.deepcopy(G1_COL_FULL)
G1_COL_FULL_SELF.spawn.articulation_props.enabled_self_collisions = True

#######################################

# ARMATURE_RS_06 = 0.012
# ARMATURE_RS_03 = 0.02
# ARMATURE_RS_00 = 0.001
# 
# NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
# DAMPING_RATIO = 2.0
# 
# STIFFNESS_RS_06 = ARMATURE_RS_06 * NATURAL_FREQ**2
# STIFFNESS_RS_03 = ARMATURE_RS_03 * NATURAL_FREQ**2
# STIFFNESS_RS_00 = ARMATURE_RS_00 * NATURAL_FREQ**2
# 
# DAMPING_RS_06 = 2.0 * DAMPING_RATIO * ARMATURE_RS_06 * NATURAL_FREQ
# DAMPING_RS_03 = 2.0 * DAMPING_RATIO * ARMATURE_RS_03 * NATURAL_FREQ
# DAMPING_RS_00 = 2.0 * DAMPING_RATIO * ARMATURE_RS_00 * NATURAL_FREQ

# Armature values from y1_v2.xml
ARMATURE_WAIST_YAW = 0.0576489375
ARMATURE_HIP_PITCH = 0.031752
ARMATURE_HIP_ROLL = 0.0576489375
ARMATURE_HIP_YAW = 0.0576489375
ARMATURE_KNEE = 0.065
ARMATURE_ANKLE_PITCH = 0.023328
ARMATURE_ANKLE_ROLL = 0.023328
ARMATURE_SHOULDER_PITCH = 0.032
ARMATURE_SHOULDER_ROLL = 0.032
ARMATURE_SHOULDER_YAW = 0.002
ARMATURE_ELBOW = 0.032
ARMATURE_WRIST_ROLL = 0.0007
ARMATURE_WRIST_PITCH = 0.0007
ARMATURE_WRIST_YAW = 0.0007

ARMATURE_FOR_PD_WAIST_YAW = 0.5 * 0.0576489375
ARMATURE_FOR_PD_HIP_PITCH = 0.5 * 0.031752
ARMATURE_FOR_PD_HIP_ROLL = 0.5 * 0.0576489375
ARMATURE_FOR_PD_HIP_YAW = 0.5 * 0.0576489375
ARMATURE_FOR_PD_KNEE = 0.5 * 0.065
ARMATURE_FOR_PD_ANKLE_PITCH = 0.5 * 0.023328
ARMATURE_FOR_PD_ANKLE_ROLL = 0.5 * 0.023328
ARMATURE_FOR_PD_SHOULDER_PITCH = 0.5 * 0.032
ARMATURE_FOR_PD_SHOULDER_ROLL = 0.5 * 0.032
ARMATURE_FOR_PD_SHOULDER_YAW = 0.002
ARMATURE_FOR_PD_ELBOW = 0.5 * 0.032
ARMATURE_FOR_PD_WRIST_ROLL = 0.0007
ARMATURE_FOR_PD_WRIST_PITCH = 0.0007
ARMATURE_FOR_PD_WRIST_YAW = 0.0007

NATURAL_FREQ = 5 * 2.0 * 3.1415926535  # 5Hz
NATURAL_FREQ_FOR_HIP_PITCH = 3.5 * 2.0 * 3.1415926535  # 3.5Hz
NATURAL_FREQ_FOR_KNEE = 3.5 * 2.0 * 3.1415926535  # 3.5Hz
NATURAL_FREQ_FOR_ANKLE = 15 * 2.0 * 3.1415926535  # 15Hz
DAMPING_RATIO = 2.

# Stiffness and damping calculated from armature values
STIFFNESS_WAIST_YAW = ARMATURE_FOR_PD_WAIST_YAW * NATURAL_FREQ**2
STIFFNESS_HIP_PITCH = ARMATURE_FOR_PD_HIP_PITCH * NATURAL_FREQ_FOR_HIP_PITCH**2
STIFFNESS_HIP_ROLL = ARMATURE_FOR_PD_HIP_ROLL * NATURAL_FREQ**2
STIFFNESS_HIP_YAW = ARMATURE_FOR_PD_HIP_YAW * NATURAL_FREQ**2
STIFFNESS_KNEE = ARMATURE_FOR_PD_KNEE * NATURAL_FREQ_FOR_KNEE**2
STIFFNESS_ANKLE_PITCH = ARMATURE_FOR_PD_ANKLE_PITCH * NATURAL_FREQ_FOR_ANKLE**2
STIFFNESS_ANKLE_ROLL = ARMATURE_FOR_PD_ANKLE_ROLL * NATURAL_FREQ_FOR_ANKLE**2
STIFFNESS_SHOULDER_PITCH = ARMATURE_FOR_PD_SHOULDER_PITCH * NATURAL_FREQ**2
STIFFNESS_SHOULDER_ROLL = ARMATURE_FOR_PD_SHOULDER_ROLL * NATURAL_FREQ**2
STIFFNESS_SHOULDER_YAW = ARMATURE_FOR_PD_SHOULDER_YAW * NATURAL_FREQ**2
STIFFNESS_ELBOW = ARMATURE_FOR_PD_ELBOW * NATURAL_FREQ**2
STIFFNESS_WRIST_ROLL = ARMATURE_FOR_PD_WRIST_ROLL * NATURAL_FREQ**2
STIFFNESS_WRIST_PITCH = ARMATURE_FOR_PD_WRIST_PITCH * NATURAL_FREQ**2
STIFFNESS_WRIST_YAW = ARMATURE_FOR_PD_WRIST_YAW * NATURAL_FREQ**2

DAMPING_WAIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_WAIST_YAW * NATURAL_FREQ
DAMPING_HIP_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_HIP_PITCH * NATURAL_FREQ_FOR_HIP_PITCH
DAMPING_HIP_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_HIP_ROLL * NATURAL_FREQ
DAMPING_HIP_YAW = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_HIP_YAW * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_KNEE * NATURAL_FREQ_FOR_KNEE
DAMPING_ANKLE_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_ANKLE_PITCH * NATURAL_FREQ_FOR_ANKLE
DAMPING_ANKLE_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_ANKLE_ROLL * NATURAL_FREQ_FOR_ANKLE
DAMPING_SHOULDER_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_SHOULDER_PITCH * NATURAL_FREQ
DAMPING_SHOULDER_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_SHOULDER_ROLL * NATURAL_FREQ
DAMPING_SHOULDER_YAW = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_SHOULDER_YAW * NATURAL_FREQ
DAMPING_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_ELBOW * NATURAL_FREQ
DAMPING_WRIST_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_WRIST_ROLL * NATURAL_FREQ
DAMPING_WRIST_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_WRIST_PITCH * NATURAL_FREQ
DAMPING_WRIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_FOR_PD_WRIST_YAW * NATURAL_FREQ

Y1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/Y1_v2/y1_v2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),

    # TO BE CHANGED
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            'waist_yaw_joint': 0.,
            
            'left_hip_pitch_joint': 0.0,
            'left_hip_roll_joint': 0.,
            'left_hip_yaw_joint': 0.,
            'left_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.0,
            'left_ankle_roll_joint': 0.0,

            'right_hip_pitch_joint': 0.0,
            'right_hip_roll_joint': 0.,
            'right_hip_yaw_joint': 0.,
            'right_knee_joint': 0.0,
            'right_ankle_pitch_joint': 0.0,
            'right_ankle_roll_joint': 0.0,

            'left_shoulder_pitch_joint': 0.,
            'left_shoulder_roll_joint': 0.,
            'left_shoulder_yaw_joint': 0.,
            'left_elbow_joint': 1.55,
            'left_wrist_pitch_joint': 0.,
            
            'right_shoulder_pitch_joint': 0.,
            'right_shoulder_roll_joint': 0.,
            'right_shoulder_yaw_joint': 0.,
            'right_elbow_joint': 1.55,
            'right_wrist_pitch_joint': 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 90.0,
                ".*_hip_roll_joint": 70.0,
                ".*_hip_yaw_joint": 70.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 16.22,
                ".*_hip_roll_joint": 12.85,
                ".*_hip_yaw_joint": 12.85,
                ".*_knee_joint": 11.2,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_HIP_PITCH,
                ".*_hip_roll_joint": STIFFNESS_HIP_ROLL,
                ".*_hip_yaw_joint": STIFFNESS_HIP_YAW,
                ".*_knee_joint": STIFFNESS_KNEE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP_PITCH,
                ".*_hip_roll_joint": DAMPING_HIP_ROLL,
                ".*_hip_yaw_joint": DAMPING_HIP_YAW,
                ".*_knee_joint": DAMPING_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_FOR_PD_HIP_PITCH,
                ".*_hip_roll_joint": ARMATURE_FOR_PD_HIP_ROLL,
                ".*_hip_yaw_joint": ARMATURE_FOR_PD_HIP_YAW,
                ".*_knee_joint": ARMATURE_FOR_PD_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=7.0,
            stiffness={
                ".*_ankle_pitch_joint": STIFFNESS_ANKLE_PITCH,
                ".*_ankle_roll_joint": STIFFNESS_ANKLE_ROLL,
            },
            damping={
                ".*_ankle_pitch_joint": DAMPING_ANKLE_PITCH,
                ".*_ankle_roll_joint": DAMPING_ANKLE_ROLL,
            },
            armature={
                ".*_ankle_pitch_joint": ARMATURE_FOR_PD_ANKLE_PITCH,
                ".*_ankle_roll_joint": ARMATURE_FOR_PD_ANKLE_ROLL,
            },
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=70.0,
            velocity_limit_sim=12.85,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_WAIST_YAW,
            damping=DAMPING_WAIST_YAW,
            armature=ARMATURE_FOR_PD_WAIST_YAW,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 27.0,
                ".*_shoulder_roll_joint": 27.0,
                ".*_shoulder_yaw_joint": 12.5,
                ".*_elbow_joint": 27.0,
                ".*_wrist_roll_joint": 5.5,
                ".*_wrist_pitch_joint": 5.5,
                ".*_wrist_yaw_joint": 5.5,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 3.768,
                ".*_shoulder_roll_joint": 3.768,
                ".*_shoulder_yaw_joint": 12.5,
                ".*_elbow_joint": 3.768,
                ".*_wrist_roll_joint": 10.46,
                ".*_wrist_pitch_joint": 10.46,
                ".*_wrist_yaw_joint": 10.46,

            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_SHOULDER_PITCH,
                ".*_shoulder_roll_joint": STIFFNESS_SHOULDER_ROLL,
                ".*_shoulder_yaw_joint": STIFFNESS_SHOULDER_YAW,
                ".*_elbow_joint": STIFFNESS_ELBOW,
                ".*_wrist_yaw_joint": STIFFNESS_WRIST_YAW,
                ".*_wrist_roll_joint": STIFFNESS_WRIST_ROLL,
                ".*_wrist_pitch_joint": STIFFNESS_WRIST_PITCH,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_SHOULDER_PITCH,
                ".*_shoulder_roll_joint": DAMPING_SHOULDER_ROLL,
                ".*_shoulder_yaw_joint": DAMPING_SHOULDER_YAW,
                ".*_elbow_joint": DAMPING_ELBOW,
                ".*_wrist_yaw_joint": DAMPING_WRIST_YAW,
                ".*_wrist_roll_joint": DAMPING_WRIST_ROLL,
                ".*_wrist_pitch_joint": DAMPING_WRIST_PITCH,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_FOR_PD_SHOULDER_PITCH,
                ".*_shoulder_roll_joint": ARMATURE_FOR_PD_SHOULDER_ROLL,
                ".*_shoulder_yaw_joint": ARMATURE_FOR_PD_SHOULDER_YAW,
                ".*_elbow_joint": ARMATURE_FOR_PD_ELBOW,
                ".*_wrist_yaw_joint": ARMATURE_FOR_PD_WRIST_YAW,
                ".*_wrist_roll_joint": ARMATURE_FOR_PD_WRIST_ROLL,
                ".*_wrist_pitch_joint": ARMATURE_FOR_PD_WRIST_PITCH,
            },
        ),
    },
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
        "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "base_link": "base_link",
        "waist_link": "waist_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
    })
)

Y1_COL_SELF = copy.deepcopy(Y1_CFG)
Y1_COL_SELF.spawn.articulation_props.enabled_self_collisions = True
