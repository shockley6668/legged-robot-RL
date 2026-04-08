import sys

filename = "configs/tinker_constraint_him_trot.py"
with open(filename, "r") as f:
    text = f.read()

# Replace command_dead with -1.0 so that is_moving_cmd is ALWAYS True, disabling ALL stand penalties beautifully
text = text.replace("command_dead = 0.05", "command_dead = -1.0  # ALWAYS evaluate as moving to disable stand_still logic")

with open(filename, "w") as f:
    f.write(text)

