# Tinker 机器人RL训练 - 快速参考指南

## 1. 项目结构速查

```
OmniBotCtrl/
├── train.py                     # 训练入口脚本
├── play.py                      # 推理/测试脚本
├── global_config.py             # 全局配置选择
│
├── configs/                     # 环境配置文件
│   ├── tinker_constraint_him_trot.py
│   ├── tinker_constraint_him_stand.py
│   └── legged_robot_config.py   # 基础配置类
│
├── envs/                        # 环境实现
│   ├── legged_robot.py          # 主环境类 (2340行)
│   ├── base_task.py             # 基础任务类
│   └── vec_env.py               # 向量化环境
│
├── algorithm/                   # 算法实现
│   └── np3o.py                  # 约束PPO算法 (298行)
│
├── runner/                      # 训练循环
│   ├── on_constraint_policy_runner.py  # 主训练器 (340行)
│   └── rollout_storage.py       # 轨迹存储
│
├── modules/                     # 神经网络模块
│   ├── actor_critic.py          # ActorCritic网络 (1949行)
│   ├── common_modules.py        # 编码器、MLP等
│   └── transformer_modules.py   # Transformer组件
│
├── utils/                       # 工具函数
│   ├── helpers.py               # 助手函数
│   ├── terrain.py               # 地形生成
│   └── task_registry.py         # 任务注册
│
└── resources/                   # URDF和资源文件
    └── TinkerV2_URDF/
        └── urdf/TinkerV2_URDF.urdf
```

---

## 2. 启动训练

### 2.1 命令行快速启动

```bash
cd /home/shockley/OmniBotSeries-Tinker/OmniBotCtrl/OmniBotCtrl
python train.py
```

### 2.2 配置项选择

**编辑** `global_config.py`:
```python
ROBOT_SEL = 'Tinker'      # 选择机器人: 'Tinker', 'Taitan', 'Tinymal'
GAIT_SEL = 'Trot'         # 步态: 'Trot' 或 'Stand'
MAX_ITER = 30000          # 最大迭代次数
SAVE_DIV = 5000           # 保存间隔
```

### 2.3 监控训练

```bash
# 方式1: Tensorboard
tensorboard --logdir=./logs

# 方式2: Weights & Biases (如果启用)
wandb login
# 在 global_config.py 中设置 en_logger = True
```

---

## 3. 关键参数速查表

### 3.1 输入输出维度

| 参数 | 值 | 说明 |
|------|-----|------|
| 自由度 (DOF) | 10 | 双足 × 5关节 |
| 本体感受 | 39 | 角速度(3) + 欧拉角(3) + 命令(3) + 关节(20) + 前动作(10) |
| LIDAR 扫描 | 187 | 激光雷达点数 |
| 特权信息 | 43 | 仅训练时 |
| 历史长度 | 10 | 过去观测时步 |
| **总观测** | **659** | 完整观测维度 |
| **动作** | **10** | 10个关节的目标角度 |

### 3.2 网络架构

| 模块 | 输入维度 | 隐层 | 输出维度 |
|------|---------|------|---------|
| 扫描编码器 | 187 | [256, 128, 64] | 64 |
| 历史编码器 | 390 | RNN | 128 |
| 融合层 | 64+128+39 | - | 256 |
| Actor | 256 | [512, 256, 128] | μ(10) + σ(10) |
| Critic-R | 256 | [512, 256, 128] | 1 |
| Critic-C | 256 | [512, 256, 128] | 9 |

### 3.3 算法超参数

| 参数 | 值 | 范围 | 调优建议 |
|------|-----|------|---------|
| 学习率 | 1e-3 | [1e-4, 1e-2] | 过小收敛慢，过大不稳定 |
| PPO 裁剪 | 0.2 | [0.1, 0.3] | 越小更新越保守 |
| 熵系数 | 0.01 | [0, 0.1] | 增加探索 |
| 折扣因子 | 0.998 | [0.99, 0.9999] | 越大越看重长期 |
| GAE λ | 0.95 | [0.9, 0.99] | 平衡偏差/方差 |
| 成本违反系数 | 1.0 | [0.1, 10] | 越大约束越严格 |
| 成本价值系数 | 0.1 | [0.01, 1] | 权衡成本学习 |

### 3.4 成本约束限值

| 约束ID | 约束名称 | 限值 (d) | 单位 |
|--------|---------|---------|------|
| 1 | torque_limit | 45 | N·m |
| 2 | pos_limit | 0.9 | 标准化 |
| 3 | dof_vel_limits | 2.4 | rad/s |
| 4 | vel_smoothness | 1.0 | - |
| 5 | acc_smoothness | 0.5 | - |
| 6 | collision | 0 | - |
| 7 | feet_contact_forces | 120 | N |
| 8 | stumble | 0.1 | - |
| 9 | foot_regular / trot_contact | 1.0 | - |

---

## 4. 观测构成详解

### 4.1 本体感受 (39维)

```python
[
  ω_x, ω_y, ω_z,                    # 基座角速度 (3)
  φ, θ, ψ,                           # 欧拉角 (Roll, Pitch, Yaw) (3)
  v_x_cmd, v_y_cmd, ω_z_cmd,         # 速度命令 (3)
  θ_L0-θ_L4, θ_R0-θ_R4,             # 关节相对位置 (10)
  ω̇_L0-ω̇_L4, ω̇_R0-ω̇_R4,           # 关节角速度 (10)
  a_{t-1}[0:10]                      # 前一时刻动作 (10)
]  # 总计 39维
```

### 4.2 特权信息 (43维，仅训练)

```python
[
  v_x, v_y, v_z,                     # 基座线速度 (3)
  contact_L, contact_R,              # 接触状态 (2)
  delay_L0-L4, delay_R0-R4,          # 传感器延迟 (10)
  mass_L0-L4, mass_R0-R4,            # 质量参数 (10)
  friction_L0-L4, friction_R0-R4,    # 摩擦系数 (10)
  kp_factor_L0-L4, kp_factor_R0-R4,  # Kp增益 (10)
  kd_factor_L0-L4, kd_factor_R0-R4   # Kd增益 (10)
]  # 总计需要根据配置 (典型 43-50)
```

### 4.3 LIDAR 扫描 (187维)

```python
# 前向二维激光扫描
scan_distances = [d_1, d_2, ..., d_187]  # 每个激光束的距离
# 扫描范围: ±90°, 最大距离: 5m
```

### 4.4 历史观测 (390维)

```python
# 过去10个时刻的本体感受
obs_history = [
  obs_t-9,  # 最久远的观测
  obs_t-8,
  ...
  obs_t-1,  # 最近的过去观测
  obs_t     # 当前观测 (不在历史中)
]
# 维度: 10 × 39 = 390
```

---

## 5. 奖励函数编辑快速指南

### 5.1 添加新的奖励项

**步骤 1**: 在环境中实现计算函数

在 `legged_robot.py` 中：
```python
def _reward_new_term(self):
    """计算新奖励项"""
    # 返回形状 (num_envs,) 的张量
    return my_reward_value  # torch.Tensor
```

**步骤 2**: 在配置中添加权重

在 `tinker_constraint_him_trot.py` 中：
```python
class rewards(LeggedRobotCfg.rewards):
    class scales(LeggedRobotCfg.rewards.scales):
        new_term = 1.0  # 权重值
```

**步骤 3**: 在 `_prepare_reward_function()` 中注册

```python
self.reward_functions = [
    # ... 其他函数 ...
    self._reward_new_term
]
```

### 5.2 常见奖励模式

**距离最小化**:
```python
def _reward_pos_error(self):
    return -torch.norm(target - current, dim=-1)
```

**范围限制**:
```python
def _reward_bounded(self):
    return torch.clamp(value, min_val, max_val)
```

**二进制奖励**:
```python
def _reward_contact(self):
    return 1.0 * (contact_force > threshold)
```

---

## 6. 成本函数编辑快速指南

### 6.1 添加新的成本约束

**步骤 1**: 在环境中实现计算函数

```python
def _cost_new_constraint(self):
    """计算新成本/约束"""
    # 返回形状 (num_envs,) 的张量
    return cost_value
```

**步骤 2**: 在配置中设置限值

```python
class cost(LeggedRobotCfg.cost):
    num_costs = 10  # 从9增加到10
    
    # 在某处定义限值字典
    d_values = {
        'torque_limit': 45,
        'pos_limit': 0.9,
        # ... 其他约束 ...
        'new_constraint': 10.0  # 新约束的限值
    }
```

**步骤 3**: 注册约束

在 `_prepare_cost_function()` 中，系统自动扫描所有 `_cost_*` 方法。

### 6.2 调试约束满足

```python
# 查看运行时的成本值
for name, sum_val in self.cost_episode_sums.items():
    print(f"{name}: {sum_val.mean():.4f} / {self.d_values[name]}")
```

---

## 7. 训练过程监控

### 7.1 关键指标

```python
# 在日志中查看这些指标
{
  'episode/rew_tracking_lin_vel': 2.5,    # 线速度追踪
  'episode/rew_tracking_ang_vel': 1.2,    # 角速度追踪
  'episode/cost_torque_limit': 2.3,       # 力矩违反
  'policy/entropy': 2.8,                  # 策略熵度
  'train/kl_divergence': 0.012,           # KL散度
  'train/value_loss': 0.15                # 价值损失
}
```

### 7.2 何时停止训练

- **回报饱和**: 平均回报不再增加超过5个checkpoint
- **约束稳定**: 所有约束连续满足10000步
- **KL散度稳定**: KL在 $[0.01, 0.1]$ 范围内

### 7.3 保存和加载模型

```bash
# 自动保存到 logs/ 目录
# 找到最好的检查点:
ls -lt logs/*/checkpoints/

# 手动加载模型进行测试
# 见下一节
```

---

## 8. 模型推理 (Deployment)

### 8.1 运行训练的模型

**编辑** `play.py`:
```python
# 选择要加载的模型路径
CHECKPOINT_PATH = '/home/shockley/OmniBotSeries-Tinker/OmniBotCtrl/OmniBotCtrl/logs/Tinker/checkpoints/best_model.pt'

# 运行推理
python play.py
```

### 8.2 模型转换

**转换为 ONNX**:
```bash
python pt2onnx.py --model modelt.pt --output policy.onnx
```

**转换为 TVM**:
```bash
python pt2tvm.py --model modelt.pt --output policy.so
```

### 8.3 Sim2Real 部署

```python
# 1. 移除特权信息编码
policy_deployment = policy.without_privilege_encoder()

# 2. 使用 LIDAR + IMU 数据
obs_real = [
  lidar_scan,          # 187维
  imu_angular_vel,     # 3维
  joint_positions,     # 10维
  joint_velocities,    # 10维
  commands             # 3维
]

# 3. 推理
action = policy_deployment(obs_real)

# 4. 转换为扭矩命令
tau = K_p * (target_angle - current_angle) - K_d * current_velocity
```

---

## 9. 常见问题排查

### 问题 1: 训练不收敛

**症状**: 回报在原地踏步不增加

**排查步骤**:
```python
# 1. 检查奖励值范围
print(f"Reward range: {rewards.min():.2f} ~ {rewards.max():.2f}")

# 2. 检查优势是否过大
print(f"Advantage range: {advantages.min():.2f} ~ {advantages.max():.2f}")

# 3. 检查学习率
if kl_div > 0.1:
    learning_rate *= 0.5
```

**解决方案**: 
- 调小学习率
- 检查奖励函数设计
- 增加熵系数促进探索

### 问题 2: 成本约束未满足

**症状**: 约束值一直超过限值

**排查步骤**:
```python
# 1. 检查约束是否可达
min_possible_cost = evaluate_minimal_strategy(env, constraint)
if min_possible_cost > d_value:
    # 约束过紧，无法满足
    d_value *= 1.5

# 2. 检查拉格朗日乘子
print(f"Lagrangian multiplier: {k_value}")

# 3. 检查是否收敛早期就强制约束
if iteration < 5000:
    k_value = base_k_value * (iteration / 5000)
```

### 问题 3: GPU 内存溢出

**症状**: CUDA out of memory 错误

**解决方案**:
```python
# 减少环境数
num_envs = 512  # 从 1024 减少

# 减少轨迹长度
num_steps_per_env = 512  # 从 1024 减少

# 使用梯度累积而不是增加 batch size
accumulation_steps = 4
batch_size = 256
```

---

## 10. 性能基准

### 10.1 预期训练时间

| 配置 | GPU | 时间 | 迭代次数 |
|------|-----|------|---------|
| 完整 | V100 | ~20h | 30000 |
| 完整 | A100 | ~8h | 30000 |
| 轻量 | V100 | ~5h | 5000 |

### 10.2 推理性能

| 模型 | FPS | 延迟 (ms) | 内存 (MB) |
|------|-----|----------|----------|
| PyTorch | 50 | 20 | 850 |
| ONNX | 80 | 12.5 | 420 |
| TVM | 100 | 10 | 350 |

---

## 11. 文件修改清单

**添加新环境约束**:
- [ ] 实现 `_cost_*()` 方法
- [ ] 更新 `cost` 配置中的 `num_costs`
- [ ] 设置约束限值 `d_values`

**调整奖励函数**:
- [ ] 修改 `scales` 中的权重
- [ ] 测试新权重的效果
- [ ] 记录模型性能变化

**改进网络结构**:
- [ ] 修改 `scan_encoder_dims`
- [ ] 调整 `actor_hidden_dims`
- [ ] 修改 `critic_hidden_dims`

---

## 12. 有用的命令

```bash
# 查看训练日志
tail -f logs/Tinker/*/events.out.tfevents.*

# 找到最佳模型
find logs -name "*.pt" -type f | xargs ls -lt | head -1

# 计算总参数量
grep -r "parameter" logs/Tinker/*/config.txt

# 清理旧训练日志
find logs -name "*.pt" -mtime +7 -delete  # 删除7天前的模型

# GPU 监控
watch -n 1 nvidia-smi
```

---

**快速参考版本**: 1.0  
**最后更新**: 2025年11月14日  
**维护者**: OmniBotSeries-Tinker 项目

