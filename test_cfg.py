import isaacgym
from configs.tinker_constraint_him_trot import TinkerConstraintHimRoughCfg
from utils.helpers import class_to_dict
env_cfg = TinkerConstraintHimRoughCfg()
scales = class_to_dict(env_cfg.rewards.scales)
print(list(scales.keys()))
