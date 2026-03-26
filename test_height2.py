import mujoco
m = mujoco.MjModel.from_xml_path('resources/TinkerV2_URDF/urdf/TinkerV2_URDF.urdf')
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)

# find joints
joints = ['J_L0', 'J_L1', 'J_L2', 'J_L3', 'J_L4_ankle', 'J_R0', 'J_R1', 'J_R2', 'J_R3', 'J_R4_ankle']
targets = [0.0, 0.08, 0.56, -1.12, -0.57, 0.0, -0.08, -0.56, 1.12, 0.57]

for j_name, tgt in zip(joints, targets):
    j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    adr = m.jnt_qposadr[j_id]
    d.qpos[adr] = tgt

mujoco.mj_kinematics(m, d)

base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
base_pos = d.xpos[base_id]

# Let's find foot bodies
foot_bodies = ["foot_l_link", "foot_r_link"] # Need to check actual names
for b in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if name and ("foot" in name.lower() or "ankle" in name.lower()):
        print(f"{name}: Z = {d.xpos[b][2]}")

