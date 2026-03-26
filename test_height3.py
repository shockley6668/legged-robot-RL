import mujoco
m = mujoco.MjModel.from_xml_path('resources/TinkerV2_URDF/urdf/TinkerV2_URDF.urdf')
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)

joints = ['J_L0', 'J_L1', 'J_L2', 'J_L3', 'J_L4_ankle', 'J_R0', 'J_R1', 'J_R2', 'J_R3', 'J_R4_ankle']
targets = [0.0, 0.08, 0.56, -1.12, -0.57, 0.0, -0.08, -0.56, 1.12, 0.57]

for j_name, tgt in zip(joints, targets):
    j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    adr = m.jnt_qposadr[j_id]
    d.qpos[adr] = tgt

mujoco.mj_kinematics(m, d)

min_z = 0

for g in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    if not name:
        name = "unnamed"
    
    # We just want the lowest point of any geom in the legs
    pos = d.geom_xpos[g]
    size = m.geom_size[g]
    type = m.geom_type[g]
    
    bottom_z = pos[2]
    # rough approximation based on sphere or capsule/cylinder
    if type == mujoco.mjtGeom.mjGEOM_SPHERE:
        bottom_z -= size[0]
    elif type == mujoco.mjtGeom.mjGEOM_CYLINDER or type == mujoco.mjtGeom.mjGEOM_CAPSULE:
         # size is roughly [radius, halflength]. usually Z axis or we can just subtract radius/halflengths
         # let's just make it simple if it's the foot
         pass
         
    if pos[2] < min_z:
        min_z = pos[2]

print(f"Lowest geom center Z = {min_z}")

# print base height again
base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
print(f"Base link Z = {d.xpos[base_id][2]}")

