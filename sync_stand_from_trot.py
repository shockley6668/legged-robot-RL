import re

with open("configs/tinker_constraint_him_trot.py", "r") as f:
    text = f.read()

# 1. Zero commands, 100% stop
text = re.sub(r"lin_vel_x\s*=\s*\[-0\.8,\s*0\.8\]", "lin_vel_x = [0.0, 0.0]", text)
text = re.sub(r"lin_vel_y\s*=\s*\[-0\.6,\s*0\.6\]", "lin_vel_y = [0.0, 0.0]", text)
text = re.sub(r"ang_vel_yaw\s*=\s*\[-0\.8,\s*0\.8\]", "ang_vel_yaw = [0.0, 0.0]", text)
text = re.sub(r"stop_rate\s*=\s*0\.8", "stop_rate = 1.0", text)

# 2. Deadzone 恢复为正常触发值
text = re.sub(r"command_dead\s*=\s*-1\.0[^\n]*", "command_dead = 0.05", text)

# 3. 超大推力和扰动
text = re.sub(r"max_push_vel_xy\s*=\s*0\.7", "max_push_vel_xy = 2.0", text)
text = re.sub(r"max_push_ang_vel\s*=\s*0\.6", "max_push_ang_vel = 1.5", text)

# 4. 去掉被注释的原地惩罚并重新激活
text = re.sub(r"\s*# # 【关闭原地站桩惩罚】.*?(?=orientation_eular)", 
              "\n            stand_still = -0.5\n            stand_still_force = -0.5\n            stand_still_step_punish = -1.0\n            base_stability = -0.5\n            \n            ", 
              text, flags=re.DOTALL)

# 5. Env names 修改名字以防重叠
text = re.sub(r"run_name\s*=\s*'test_barlowtwins_phase2'", "run_name = 'test_stand_push'", text)
text = re.sub(r"run_name\s*=\s*'test_barlowtwins'", "run_name = 'test_stand_push'", text)

# 写入
with open("configs/tinker_constraint_him_stand.py", "w") as f:
    f.write(text)

