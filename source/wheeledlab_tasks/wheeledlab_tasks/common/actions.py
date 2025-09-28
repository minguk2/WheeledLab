from isaaclab.utils import configclass

from wheeledlab.envs.mdp import RCCar4WDActionCfg, RCCarRWDActionCfg

@configclass
class F1Tenth4WDActionCfg:
    """Action configuration for F1Tenth 4WD, using RCCar4WDActionCfg with F1Tenth's joint names."""
    throttle_steer = RCCar4WDActionCfg(
        wheel_joint_names=[
            "wheel_back_left",
            "wheel_back_right",
            "wheel_front_left",
            "wheel_front_right",
        ],
        steering_joint_names=[
            "rotator_left",
            "rotator_right",
        ],
        base_length=0.365,
        base_width=0.284,
        wheel_radius=0.05,
        scale=(3.0, 0.488),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )

