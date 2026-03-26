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

# Find feet
left_foot = None
right_foot = None

for b in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if not name: continue
    if "l4_link_ankle" in name.lower() or "foot_l" in name.lower() or "l4" in name.lower() and "ankle" in name.lower():
        left_foot = d.xpos[b]
    if "r4_link_ankle" in name.lower() or "foot_r" in name.lower() or "r4" in name.lower() and "ankle" in name.lower():
        right_foot = d.xpos[b]

print(f"Left foot pos: {left_foot}")
print(f"Right foot pos: {right_foot}")
if left_foot is not None and right_foot is not None:
    print(f"Y distance between feet: {abs(left_foot[1] - right_foot[1])}")
    print(f"Base Z height: {-min(left_foot[2], right_foot[2])}")
