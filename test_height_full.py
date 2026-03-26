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

min_z = 0

for g in range(m.ngeom):
    pos = d.geom_xpos[g]
    size = m.geom_size[g]
    type = m.geom_type[g]
    mat = d.geom_xmat[g] # 3x3 rotation matrix, flattened as 9 elements
    
    # Simple bounding box approximation
    if type == mujoco.mjtGeom.mjGEOM_SPHERE:
        bottom_z = pos[2] - size[0]
    elif type == mujoco.mjtGeom.mjGEOM_CYLINDER or type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # size is roughly [radius, hz]
        # In URDF, capsules/cylinders are symmetric around local Z axis
        # geom_xmat[8] is the local Z axis projected to world Z
        z_direction = abs(mat[8]) 
        bottom_z = pos[2] - (size[1] * z_direction + size[0])
    elif type == mujoco.mjtGeom.mjGEOM_BOX:
        # half sizes: size[0], size[1], size[2]
        z_offset = abs(mat[2])*size[0] + abs(mat[5])*size[1] + abs(mat[8])*size[2]
        bottom_z = pos[2] - z_offset
    else:
        bottom_z = pos[2] # Fallback
         
    if bottom_z < min_z:
        min_z = bottom_z

print(f"Lowest point Z (including geometry radius) = {min_z}")
print(f"To stand perfectly, base_height_target should be: {-min_z}")
