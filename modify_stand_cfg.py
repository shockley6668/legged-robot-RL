import sys

filename = "configs/tinker_constraint_him_stand.py"
with open(filename, "r") as f:
    text = f.read()

# Commands configuration
text = text.replace("lin_vel_x = [-0.2, 0.2]", "lin_vel_x = [0.0, 0.0]")
text = text.replace("lin_vel_y = [-0.2, 0.2]", "lin_vel_y = [0.0, 0.0]")
text = text.replace("ang_vel_yaw = [-1.0, 1.0]", "ang_vel_yaw = [0.0, 0.0]")
text = text.replace("stop_rate = 0.25", "stop_rate = 1.0")

# Push force configuration
text = text.replace("push_interval_s = 6", "push_interval_s = 4.0")
text = text.replace("max_push_vel_xy = 0.7", "max_push_vel_xy = 2.0")  # Huge push!
text = text.replace("max_push_ang_vel = 0.6", "max_push_ang_vel = 1.5") # Huge ang push!

# Reward scales
# Enable basic standing penalties in this config
text = text.replace("stand_still_force = -0.1", "stand_still_force = -0.5\n            stand_still_step_punish = -0.5")

with open(filename, "w") as f:
    f.write(text)

