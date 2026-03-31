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
            'J_L1':  0.08,   # [rad]
            'J_L2':  0.56,   # [rad]
            'J_L3':  -1.12,   # [rad]
            'J_L4_ankle': -0.57,   # [rad]

            'J_R0':   0.0,   # [rad]
            'J_R1':  -0.08,   # [rad]
            'J_R2':  -0.56,   # [rad]
            'J_R3':  1.12,   # [rad]
            'J_R4_ankle': 0.57,   # [rad] 
        }

        default_joint_angles_st = { # UNIFIED: same as default_joint_angles for consistency
            'J_L0':   0.0,   # [rad]
            'J_L1':  0.08,   # [rad]  Same as walking pose
            'J_L2':  0.56,   # [rad]
            'J_L3':  -1.12,   # [rad]
            'J_L4_ankle': -0.57,   # [rad]

            'J_R0':   0.0,   # [rad]
            'J_R1':  -0.08,   # [rad]  Same as walking pose
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
        stiffness = {'J_L0': 15, 'J_L1': 15,'J_L2': 15, 'J_L3':15, 'J_L4_ankle':13,
                     'J_R0': 15, 'J_R1': 15,'J_R2': 15, 'J_R3':15, 'J_R4_ankle':13}
        damping = {'J_L0': 0.3, 'J_L1': 0.65,'J_L2': 0.65, 'J_L3':0.65, 'J_L4_ankle':0.3,
                   'J_R0': 0.3, 'J_R1': 0.65,'J_R2': 0.65, 'J_R3':0.65, 'J_R4_ankle':0.3}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
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
            lin_vel_x = [-0.8, 0.8]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]  # 降低侧向平移范围，防止为了走大速度而劈叉
            ang_vel_yaw = [-0.8, 0.8]  # 之前是 [0.5, 0.5] 导致机器人根本没学过直行和左转的区别
            
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
        clearance_height_target = -0.25 # 【修改目标离地间隙】：之前-0.21要求相对于重心抬升高达12cm，现改为-0.25(抬升8cm左右)，降动作幅度
        tracking_sigma = 0.1 # 【恢复为正常范围0.1，官方默认0.25】：极端的0.001会导致网络因得不到正向梯度而放弃学习，直接摆烂。

        cycle_time = 0.4 # 作为兜底参数
        cycle_time_range = [0.45, 0.25] # 核心：低速时单腿挥动0.45s(全步态0.9s，走得慢且稳)，高速时单腿挥动0.25s(全步态0.5s，高频步态)
        touch_thr = 10.0 #N 【修改】：约1kg承重。双足机器人走路时，脚底必须受力超过1kg才能认定为踏实地面，避免微小蹭地欺骗判定
        command_dead = 0.05  # INCREASED from 0.01 to 0.05 - larger dead zone for better zero-velocity control
        stop_rate = 0.7  # ENHANCED: Increased from 0.5 (50% zero-velocity training for better standing)
        target_joint_pos_scale = 0.17    # rad
        
        max_contact_force = 140.0 #N 【大幅放宽】：机器人自重约9kg(单腿静止45N)，考虑迈步时的2-3倍动量冲击，承受150N是正常的，超150N才算砸地
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -20.0
            tracking_lin_vel = 20.0     # 提高跟速奖励，对抗平滑惩罚，解决低速不走的问题
            tracking_ang_vel = 20.0     # 适当提高以保证低速也能响应
            base_acc = 0.02
            lin_vel_z = 0.0
            ang_vel_xy = -0.05
            base_height = 2.0        # 增加机器人的高度奖励权重，逼迫其站起来
            
            collision = 0.0
            feet_stumble = 0.0
            action_rate = -0.6   # 【加重】：更加限制高频颤动现象，迫使动作更柔和
            # action_smoothness=-0.01
            # energy
            powers = -5e-6           # Reduced penalty for more active movement
            action_smoothness = -1.0 # 【恢复】：解除极端的动作平滑惩罚，以恢复低速踏步的动作意愿
            torques = -8e-6          # 不要惩罚太大，否则机器人会为了省力矩而无法支撑体重（导致下蹲劈叉）
            dof_vel = -0.005         # 【重新加重】：因为已经添加了死区(3.0rad/s)，超出部分一定是“爆转”，给予一定程度的惩罚
            dof_acc = -5e-5          # 【重新加重】：因为已经添加了死区(60rad/s2)，超出部分是“爆冲”，给予惩罚
            
            # Limit Violations (Start Penalizing)
            dof_pos_limits = -10.0
            torque_limits = -0.1
            
            # ADJUSTED penalties for standing still
            stand_still = -8.0              # 【更加重惩罚】彻底解决站立时脚1前1后和hip歪的问题，静止时必须完美复原默认姿态！
            stand_still_force = 0.3         # Force penalty
            stand_still_step_punish = -3.0   # 严禁原地抽搐踏步
            base_stability = -2.0            # Stability penalty
            
            # --- 解决侧行卡死的关键 ---
            # 机器人没有Ankle Roll横向踝关节，侧向跨步必然导致躯干短暂侧倾。由于侧向跟速收益有限(6分)，如果躯干倾斜惩罚过高(原先15.0或5.0)，网络宁可扣掉跟速分，也绝对不敢动！
            orientation_eular= 2.5           # 【稍微调软平衡奖励】：允许一定程度的上半身微微侧倾，换取腿能够靠拢且依然能侧移。
            
            feet_air_time = 1.0             # 【大幅降低】：既然你想让它走得又轻又稳，就绝不能为了赚滞空分数而高抬腿硬踩。
            foot_clearance = -2.0 # 【大幅降低】：允许相对贴地的步态
            foot_clearance_positive = 1.0  # 完全不鼓励把腿抬太高
            stumble= -0.05
            
            no_jump = 1.7
 
            # --- 解决静止时劈叉问题的关键 ---
            hip_pos= -35.0                    # 【再加重】解决hip还是有点歪的问题，严格约束hip回正。
            #feet_rotation = 1e-1
            feet_rotation1 = 0.3
            feet_rotation2 = 0.3
            #ankle_pos = 1e-5
            
            feet_contact_forces = -10.0    # 【大倍率惩罚】：最核心惩罚！如果落地力量超过45N（刚才修改的），就会被被扣重分！这样就不会重重踏地了。
            #vel_mismatch_exp = 0.3  # lin_z; ang x,y  速度奖励大可以鼓励机器人更多移动，与摆腿耦合
            low_speed = 2.0               # 【大幅提高】：明确补贴小指令下的移动，直接解决低速不跟随、>0.5才动的问题
            track_vel_hard = 8.0
            foot_slip = -0.05


    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True 
        friction_range = [0.2, 2.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5] # Increased mass range
        randomize_base_com = True
        added_com_range = [-0.03, 0.03] # Increased COM range
        push_robots = True
        push_interval_s = 4.0             # Push more frequently for robust balance
        max_push_vel_xy = 0.5         # Stronger push force (phase 2 training)
        max_push_ang_vel = 0.5          # Angular disturbance (phase 2)
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
        
        add_dof_lag = False
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1]
        
        add_dof_pos_vel_lag = False #影响性能
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [0, 1]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [0, 1]
        
        add_imu_lag =  True
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 1]
        

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
            hip_pos = 3.0 # 【大幅加重】：拉格朗日Cost的惩罚系数，原1.5。强化不歪的要求
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
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 7e-5   # 【调低为微调速率】：从 1e-4 降为 5e-5，适合加载预训练模型(resume)后的温和改造，避免原本学会的技能被瞬间洗白
        schedule = 'fixed'      # 【强制固定】：强行维持学习率，防止因刚修改过惩罚项导致 KL 激增从而断崖式掉 LR，could be adaptive, fixed
        gamma = 0.98
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        weight_decay = 0

    class policy( LeggedRobotCfgPPO.policy):
        num_costs = 5
        init_noise_std = 1.0
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
        resume = True
        resume_path = '/home/fsr/legged-robot-RL/logs/rough_go2_constraint/Mar30_12-56-12_test_barlowtwins_phase2/model_8000.pt'