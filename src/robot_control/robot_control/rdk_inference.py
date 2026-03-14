import numpy as np
import onnxruntime as ort
from collections import deque
import time

class TinkerRealInference:
    def __init__(self, model_path):
        """
        初始化 RDK 推理类
        :param model_path: ONNX 模型路径
        """
        # 1. 初始化 ONNX Session
        # 如果在 RDK 上运行，建议后续转换为 .bin 模型使用 BPU 加速
        self.session = ort.InferenceSession(model_path)
    
        # 2. 核心配置参数 (必须与训练时一致)
        self.n_proprio = 39        # 本体感受维度
        self.n_priv_latent = 44    # 特权信息长度 (修正为 44 以满足 660 维总输入)
        self.n_scan = 187          # 扫描数据长度
        self.history_len = 10      # 历史观测长度
        self.num_observations = 660 # 总输入维度 (39 + 44 + 187 + 10*39)
        
        # 3. 归一化系数 (Scales)
        self.obs_scales_ang_vel = 0.25
        self.obs_scales_quat = 1.0
        self.obs_scales_dof_pos = 1.0
        self.obs_scales_dof_vel = 0.05
        self.obs_scales_lin_vel = 2.0  # 用于用户指令
        
        # 4. 机器人控制参数
        self.action_scale = 0.25
        # 默认关节角度 (与 sim2sim_tinker.py 一致)
        self.default_dof_pos = np.array([
            0.0, 0.08, 0.56, -1.12, -0.57,  # 左腿: Yaw, Roll, Pitch, Knee, Ankle
            0.0, -0.08, -0.56, 1.12, 0.57   # 右腿: Yaw, Roll, Pitch, Knee, Ankle
        ], dtype=np.float32)
        
        # 5. 状态缓存
        self.hist_obs = deque(maxlen=self.history_len)
        self.last_action = np.zeros(10, dtype=np.float32)
        self.action_flt = np.zeros(10, dtype=np.float32) # 动作滤波器缓存

        # 6. 硬限位保护 (根据 URDF 设置)
        self.joint_limits_low = np.array([
            -0.7, -0.38, -1.57, -2.35, -1.2,  # 左腿 L0-L4
            -0.7, -0.38, -1.57,  0.0,  -1.2   # 右腿 R0-R4
        ])
        self.joint_limits_high = np.array([
             0.7,  0.46,  1.57,  0.0,   1.2,  # 左腿
             0.7,  0.47,  1.57,  2.35,  1.2   # 右腿
        ])
        
        # 初始化历史缓冲区
        for _ in range(self.history_len):
            self.hist_obs.append(np.zeros(self.n_proprio, dtype=np.float32))

    def quaternion_to_euler(self, quat):
        """ 
        四元数转欧拉角 [roll, pitch, yaw] 
        输入 quat 格式应符合 IMU 输出 (通常为 [x, y, z, w])
        """
        x, y, z, w = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def get_action(self, euler, imu_gyro, joint_pos, joint_vel, cmd_vx, cmd_vy, cmd_dyaw):
        """
        核心推理逻辑
        :param euler: 欧拉角 [roll, pitch, yaw]
        :param imu_gyro: IMU 角速度 [wx, wy, wz] (rad/s)
        :param joint_pos: 10个关节的实际位置 (rad)
        :param joint_vel: 10个关节的实际速度 (rad/s)
        :param cmd_vx: 用户前进速度指令 (m/s)
        :param cmd_vy: 用户横移速度指令 (m/s)
        :param cmd_dyaw: 用户转向角速度指令 (rad/s)
        :return: target_q (10维关节目标位置)
        """
        # A. 构建当前时刻的本体感受观测 (39维)
        obs = np.zeros(self.n_proprio, dtype=np.float32)
        
        # [0:3] 角速度缩放
        obs[0:3] = imu_gyro * self.obs_scales_ang_vel
        # [3:6] 欧拉角缩放
        obs[3:6] = euler * self.obs_scales_quat
        # [6:9] 控制指令缩放
        obs[6] = cmd_vx * self.obs_scales_lin_vel
        obs[7] = cmd_vy * self.obs_scales_lin_vel
        obs[8] = cmd_dyaw * self.obs_scales_ang_vel
        # [9:19] 关节角度偏移缩放
        obs[9:19] = (joint_pos - self.default_dof_pos) * self.obs_scales_dof_pos
        # [19:29] 关节速度缩放
        obs[19:29] = joint_vel * self.obs_scales_dof_vel
        # [29:39] 融入上一时刻动作
        obs[29:39] = self.last_action
        
        # B. 组装输入向量 (对齐训练时的维度排列)
        # 结构: [当前帧(39) | 特权信息+扫描数据(230) | 历史数据(390)]
        policy_input = np.zeros((1, self.num_observations), dtype=np.float32)
        policy_input[0, 0:39] = obs
        # 填充历史数据 (从 deque 中取出 10 帧)
        for i in range(self.history_len):
            start_off = 39 + self.n_priv_latent + self.n_scan # 39 + 43 + 187 = 269
            policy_input[0, start_off + i*39 : start_off + (i+1)*39] = self.hist_obs[i]
            
        # C. ONNX 推理
        ort_inputs = {self.session.get_inputs()[0].name: policy_input}
        action = self.session.run(None, ort_inputs)[0][0] 
        
        # D. 动作后处理
        # 1. 裁剪动作范围
        action = np.clip(action, -1.2, 1.2)
        
        # 2. 低通滤波 (必须与 sim2sim_tinker.py 的 FIR 逻辑完全一致)
        # sim2sim: action_flt = last_actions * 0.1 + action * 0.9
        flt = 0.1
        action_filtered = self.last_action * flt + action * (1 - flt)
        
        # E. 更新状态机
        self.hist_obs.append(obs)     # 将当前帧加入历史
        self.last_action = action.copy()     # 缓存模型原始输出，供下一帧滤波器使用
        
        # F. 映射到物理量
        # 这里的 action_filtered 才是送往电机的
        target_q = action_filtered * self.action_scale + self.default_dof_pos
        
        # G. 硬限位保护
        target_q = np.clip(target_q, self.joint_limits_low, self.joint_limits_high)
        
        return target_q

# --- 独立运行测试/参考 ---
if __name__ == "__main__":
    # 配置模型路径
    MODEL_FILE = "/root/legged-robot/src/robot_control/robot_control/modelt_0201.onnx"
    
    try:
        runner = TinkerRealInference(MODEL_FILE)
        print(f"成功加载模型: {MODEL_FILE}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit(1)

    # 模拟外部控制循环 (建议频率 50Hz)
    print("开始推理循环示例...")
    # 模拟数据
    mock_quat = [0.0, 0.0, 0.0, 1.0]
    mock_gyro = [0.0, 0.0, 0.0]
    mock_j_pos = np.array([0.0, 0.08, 0.56, -1.12, -0.57, 0.0, -0.08, -0.56, 1.12, 0.57])
    mock_j_vel = np.zeros(10)
    
    for count in range(100):
        start_t = time.time()
        
        # 获取动作指令
        targets = runner.get_action(
            mock_quat, mock_gyro, mock_j_pos, mock_j_vel,
            cmd_vx=0.3, cmd_vy=0.0, cmd_dyaw=0.0
        )
        
        end_t = time.time()
        print(f"Step {count}: 推理耗时 {(end_t - start_t)*1000:.2f}ms")
        
        # 在 RDK 上这里需添加 CAN 或 SDK 发送代码
        # time.sleep(0.02) # 50Hz
