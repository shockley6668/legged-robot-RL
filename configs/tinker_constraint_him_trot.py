# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from configs.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from global_config import MAX_ITER,SAVE_DIV
class TinkerConstraintHimRoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 6144  # Increased from 4096 to 6144 to fully saturate 24GB VRAM 

        n_scan = 187
        n_priv_latent =  4 + 1 + 12 + 12 + 12 + 6 + 1 + 4 + 1 - 3 + 4 -10
        n_proprio = 39 #原始观测
        history_len = 10
        num_observations = n_proprio  + n_priv_latent  + n_scan + history_len*n_proprio
        amao=1
        num_actions = 10
        en_logger = False #wanda

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m] - 提高初始高度，防止因为太低直接压垮劈叉

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'J_L0':   0.0,   # [rad]
            'J_L1':  0.0,   # [rad]
            'J_L2':  0.56,   # [rad]
            'J_L3':  -1.12,   # [rad]
            'J_L4_ankle': -0.57,   # [rad]

            'J_R0':   0.0,   # [rad]
            'J_R1':  0.0,   # [rad]
            'J_R2':  -0.56,   # [rad]
            'J_R3':  1.12,   # [rad]
            'J_R4_ankle': 0.57,   # [rad] 
        }

        default_joint_angles_st = { # UNIFIED: same as default_joint_angles for consistency
            'J_L0':   0.0,   # [rad]
            'J_L1':  0.00,   # [rad]  Same as walking pose
            'J_L2':  0.56,   # [rad]
            'J_L3':  -1.12,   # [rad]
            'J_L4_ankle': -0.57,   # [rad]

            'J_R0':   0.0,   # [rad]
            'J_R1':  0.00,   # [rad]  Same as walking pose
            'J_R2':  -0.56,   # [rad]
            'J_R3':  1.12,   # [rad]
            'J_R4_ankle': 0.57,   # [rad] 
        }
    # stiffness = {'leg_roll': 200.0, 'leg_pitch': 350.0, 'leg_yaw': 200.0,
    #                 'knee': 350.0, 'ankle': 15}
    # damping = {'leg_roll': 10, 'leg_pitch': 10, 'leg_yaw':
    #             10, 'knee': 10, 'ankle': 10}  关节名字可以区分，mujoco下与其一致
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 10.0}  # [N*m/rad]
        # damping = {'joint': 0.4}     # [N*m*s/rad]
        # 恢复接近实机的较软刚度，依靠RL的Reward强化约束它站立
        # 方案A：J_L0/R0(effort=12Nm)和J_L4/R4(effort=12Nm)降到Kp=10，避免力矩超限被clamp
        # J_L1/L2/L3(effort=20Nm)保持Kp=15
        stiffness = {'J_L0': 15, 'J_L1': 15,'J_L2': 15, 'J_L3':15, 'J_L4_ankle':15,
                     'J_R0': 15, 'J_R1': 15,'J_R2': 15, 'J_R3':15, 'J_R4_ankle':15}
        damping = {'J_L0': 0.5, 'J_L1': 0.5,'J_L2': 0.5, 'J_L3':0.5, 'J_L4_ankle':0.5,
                   'J_R0': 0.5, 'J_R1': 0.5,'J_R2': 0.5, 'J_R3':0.5, 'J_R4_ankle':0.5}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # 保持原始值
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 1

        use_filter = True

    class lession():
        stop = True # Enable discrete stop commands (25% chance)

    class commands( LeggedRobotCfg.control ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 5  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 12  # time before command are changed[s]W
        heading_command = False  # FIXED: Disabled to prevent overriding zero-velocity commands and match inference
        global_reference = False

        class ranges:
            # Setting to 0 for pure standing training
            lin_vel_x = [-0.5, 0.5]  # 稍微缩小范围，先学稳定慢走
            lin_vel_y = [-0.4, 0.4]  # 适度降低侧移，防劈叉
            ang_vel_yaw = [-0.6, 0.6]  # 适度旋转范围
            
            # Note: 60% of envs will have stop_flag=0 (zero velocity commands)
            # 40% will use these ranges (walking commands)
            
            heading = [-3.14, 3.14]
            height = [0.12 , 0.2] # m

    class asset( LeggedRobotCfg.asset ):
        #file = '{ROOT_DIR}/resources/tinker/urdf/tinker_urdf.urdf'
        #file = '{ROOT_DIR}/resources/tinker/urdf/tinker_urdf_inv_4.urdf'
        file = '{ROOT_DIR}/resources/TinkerV2_URDF/urdf/TinkerV2_URDF.urdf'
        #file = '{ROOT_DIR}/resources/tinker/urdf/tinker_urdf_inv_2_wbx(1).urdf'
    
        foot_name = "ankle" #URDF需要具有foot的link
        name = "tinker"
        penalize_contacts_on = ["ankle"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter 自己碰撞会抽搐,同时足底力计算也不准确
        flip_visual_attachments = False #
        
    class noise:#测量噪声
        add_noise = True
        noise_level = 1.0  # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.15
            ang_vel = 0.2
            gravity = 0.07
            quat = 0.05
            height_measurements = 0.02
            contact_states = 0.05
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.29 # 【恢复0.29】：机器人的默认站立真实高度其实是0.29，强行要求0.33太高了，会导致它站不住、蹲下甚至倒下。
        clearance_height_target = -0.23 # （原-0.28）调高步伐高度（离地更高一些）
        tracking_sigma = 0.1 # 【恢复为正常范围0.1，官方默认0.25】：极端的0.001会导致网络因得不到正向梯度而放弃学习，直接摆烂。
    
        cycle_time = 0.45 # (原0.35) 调大周期时长，慢步频
        # cycle_time_range = [0.45, 0.25] # 核心：低速时单腿挥动0.45s(全步态0.9s，走得慢且稳)，高速时单腿挥动0.25s(全步态0.5s，高频步态)
        touch_thr = 10.0 #N 【修改】：约1kg承重。双足机器人走路时，脚底必须受力超过1kg才能认定为踏实地面，避免微小蹭地欺骗判定
        command_dead = 0.05  # 【核心修复】恢复正常死区，让stand_still相关reward分支生效（原-1.0完全禁用了静止判定！）
        stop_rate = 0.5  # 50%零速训练 + 50%行走训练，单模型需要均衡学习两种状态
        target_joint_pos_scale = 0.17    # rad
        
        # BDX-R 三档速度分级阈值
        walking_threshold = 0.05  # total_speed < 此值 → standing模式(全锁紧)
        running_threshold = 0.3   # total_speed >= 此值 → running模式(全放松)
        # 每关节容忍度 std (顺序: L0 yaw, L1 roll, L2 pitch, L3 knee, L4 ankle, R0-R4同左)
        std_standing = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # 站立：全锁紧
        std_walking  = [0.15, 0.05, 0.5,  0.5,  0.15, 0.15, 0.05, 0.5,  0.5,  0.15]  # 走路：放开pitch+knee
        std_running  = [0.3,  0.1,  0.8,  0.8,  0.4,  0.3,  0.1,  0.8,  0.8,  0.4]   # 跑步：进一步放松
        
        max_contact_force = 120.0 #N 2.2倍体重：正常走路不超，重踏才超→触发惩罚
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -20.0
            tracking_lin_vel = 10.0     # 适度降低：防止为了追速而前倾（tracking:orientation=10:8，前倾5°亏4.6分，跟速最多赚1.6分，绝对不划算）
            tracking_ang_vel = 8.0      # 同步降低
            base_acc = 0.02
            lin_vel_z = 0.0
            ang_vel_xy = -0.05
            base_height = 2.0        # 增加机器人的高度奖励权重，逼迫其站起来
            
            collision = 0.0
            feet_stumble = 0.0
            action_rate = -1.0   # 【加重】：更加限制高频颤动现象，迫使动作更柔和 (原-0.6)
            # action_smoothness=-0.01
            # energy
            powers = -5e-8           # Reduced penalty for more active movement
            action_smoothness = -0.5 # (原-0.2) 提升平滑度要求，动作更温柔
            torques = -8e-9          # 不要惩罚太大，否则机器人会为了省力矩而无法支撑体重（导致下蹲劈叉）
            dof_vel = -0.001         # (原-0.0005) 限制爆转
            dof_acc = -5e-6          # (原-1e-6) 限制爆冲
            
            # Limit Violations (Start Penalizing)
            dof_pos_limits = -10.0
            torque_limits = -0.1
            
            # 【BDX-R核心】三档自适应姿态约束（替代 stand_still 的主要功能）
            # 永远活跃，不依赖 is_disturbed，不存在死循环问题
            variable_posture = 3.0          # 正向奖励：姿态越标准越高分(输出0~1)
            
            # 以下作为补充信号（variable_posture 已覆盖主要功能，这些可以更轻）
            stand_still = -5              # 轻度补充：零速时关节偏离惩罚
            stand_still_force = 5         # 奖励零速时左右脚受力均匀
            stand_still_step_punish = -1.0  # 新版持续惩罚(每帧≈0.5-2)，不需要太大scale
            base_stability = -3.0           # 惩罚零速时身体晃动
            stand_2leg = 2.0                # 奖励零速时双脚着地
            
            # --- 解决侧行卡死的关键 ---
            # 机器人没有Ankle Roll横向踝关节，侧向跨步必然导致躯干短暂侧倾。由于侧向跟速收益有限(6分)，如果躯干倾斜惩罚过高(原先15.0或5.0)，网络宁可扣掉跟速分，也绝对不敢动！
            orientation_eular= 8.0           # 【防前倾核心】身体正直奖励：5°前倾亏损=(1-0.42)*8=4.6分，绝对大于前倾带来的跟速收益
            
            feet_air_time = 5.0              # 适度降低（原10.0太高会鼓励高抬腿）
            foot_clearance = -5.0 # 适度惩罚离地过高，配合-0.28贴地目标
            foot_clearance_positive = 0.0  # 关闭：正向抬脚奖励会鼓励不必要的抬腿
            stumble= -0.05
            
            no_jump = 3.0  # 步态强制器：有速度→必须踏步(单脚+2)，赖着不走→重罚(双脚-2)
 
            # --- 解决内八劈叉的关键 ---
            hip_pos =-10.0                    # 【超级加重】严打“内八”与劈叉！只要敢偏离直立跨步轨迹，直接扣重分！
            #feet_rotation = 1e-1
            feet_rotation1 = 4.0
            feet_rotation2 = 4.0
            #ankle_pos = 1e-5
            
            feet_contact_forces = -8.0    # (原-5.0) 加重砸地惩罚，逼迫轻柔落地
            vel_mismatch_exp = 2  # 【防前倾利器】惩罚Z轴线速度+Roll/Pitch角速度，前倾时这两个值会飙升→直接扣分
            low_speed = 1.5               # 适度奖励低速移动
            track_vel_hard = 4.0          # 适度降低硬跟踪
            foot_slip = -0.05


    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True 
        friction_range = [0.2, 2.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0] #  扩大质量随机区间，对抗实机自重测量误差
        randomize_base_com = True
        added_com_range = [-0.05, 0.05] # 扩大质心偏移
        push_robots = True
        push_interval_s = 5.0             # Push more frequently for robust balance
        max_push_vel_xy = 0.5         #  增强扰动速度，逼迫机器人学会“大跨步回位”
        max_push_ang_vel = 0.4          # Angular disturbance
        # dynamic randomization
        # action_delay = 0.5
        action_noise = 0.015

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]#比例系数

        randomize_kpkd = True
        kp_range = [0.8,1.2]#比例系数
        kd_range = [0.8,1.2]

        randomize_lag_timesteps = True
        lag_timesteps = 8
    
        #old mass randomize new------------------------------
        randomize_all_mass = True
        rd_mass_range = [0.9, 1.1]

        randomize_com = True #link com
        rd_com_range = [-0.02, 0.02]
    
        random_inertia = True
        inertia_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.1, 0.1] # Increased offset range (approx ±5.7 deg)
        
        # add_lag = True #action lag
        # randomize_lag_timesteps = True
        # randomize_lag_timesteps_perstep = False
        # lag_timesteps_range = [5, 40]
        
        add_dof_lag = True # [加强排查1] 实机电机反馈存在网络延迟（CAN/EtherCAT），必须开启
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = True # [核心杀手锏] 让每一步的延迟都可能抖动（模拟抖动Jitter），打断固定延迟的死记硬背
        dof_lag_timesteps_range = [0, 2] # 0~40ms 的随机延迟跳动
        
        add_dof_pos_vel_lag = False #影响性能
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [0, 1]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [0, 1]
        
        add_imu_lag =  True
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = True # [核心杀手锏] IMU数据的读取存在频率跳动
        imu_lag_timesteps_range = [0, 2] # 扩大IMU延迟范围，实机IMU滤波本身就会引入相位滞后
        

    class depth( LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True
    
    class costs:
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1
            feet_air_time = 0.1
            #acc_smoothness = 0.1
            #collision = 0.1
            #stand_still = 0.1 #站立默认位置
            hip_pos = 0.1 # 【大幅加重】：拉格朗日Cost的惩罚系数，原1.5。强化不歪的要求
            #base_height = 0.1
            #foot_regular = 0.1
            #trot_contact = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            feet_air_time = 0.1
            #acc_smoothness = 0.0
            #collision = 0.0
            #stand_still = 0.0
            hip_pos = 0.022 # 【设置微小容忍度】：给0.022的容忍空间(约0.2rad偏差)，防止因为小幅度的走动误差被拉格朗日乘子无限放大。
            #base_height = 0.0
            #foot_regular = 0.0
            #trot_contact = 1
 
    class cost:
        num_costs = 5 #需要同步修改 policy
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False

class TinkerConstraintHimRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # entropy_coef = 0.01
        # learning_rate = 1.e-3
        # max_grad_norm = 1
        # num_learning_epochs = 5
        # num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # cost_value_loss_coef = 0.1
        # cost_viol_loss_coef = 1

        # entropy_coef = 0.001
        # learning_rate = 1e-5
        # num_learning_epochs = 2
        # gamma = 0.994
        # lam = 0.9
        # num_mini_batches = 4
    
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.02  # 稍增探索，防止微调时塌缩到局部最优
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1e-4    # 微调用低LR
        schedule = 'adaptive'   # 自适应LR：让KL自动调节
        gamma = 0.98
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        weight_decay = 0

    class policy( LeggedRobotCfgPPO.policy):
        num_costs = 5
        init_noise_std = 1.0  # 微调保持标准噪声
        continue_from_last_std = True
        scan_encoder_dims = None#[128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False

        teacher_act = False
        imi_flag = False
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'test_barlowtwins_phase2'
        experiment_name = 'rough_go2_constraint'
        policy_class_name = 'ActorCriticMixedBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = MAX_ITER #最大训练回合
        save_interval = SAVE_DIV #保存周期
        num_steps_per_env = 24
        resume = True  # 基于modelt.pt微调（从头训练失败：惩罚过重导致策略塌缩）
        resume_path = '/home/fsr/legged-robot-RL/logs/rough_go2_constraint/Apr21_21-34-31_test_barlowtwins_phase2/model_4600.pt'