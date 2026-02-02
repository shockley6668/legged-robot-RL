# Tinker 双足机器人强化学习训练理论文档

## 1. 概述

本文档详细阐述 OmniBotSeries-Tinker 项目中双足机器人策略学习的理论框架。该系统采用**约束强化学习 (Constrained RL)** 范式，通过 **NP3O (Neural Predictive Policy with Optimization and Objectives)** 算法实现机器人的步行控制。

---

## 2. 自由度分析

### 2.1 机器人运动学自由度

**Tinker 双足机器人** 具有以下关节结构：

#### 2.1.1 关节配置

| 关节名称 | 数量 | 功能 | 活动范围 |
|---------|------|------|---------|
| J_L0 (左髋横向) | 1 | 腿部横向摆动 | ±π rad |
| J_L1 (左髋前后) | 1 | 腿部前后摆动 | ±0.5π rad |
| J_L2 (左膝) | 1 | 大腿小腿相对运动 | 0~1.2π rad |
| J_L3 (左踝前后) | 1 | 足部前后灵活性 | ±0.5π rad |
| J_L4_ankle (左踝翻滚) | 1 | 足部翻滚 | ±0.6π rad |
| 右侧镜像关节 | 5 | 对称结构 | 对应左侧 |

**总自由度 (DOF)**: **10 DOF** (双足 × 5个关节)

#### 2.1.2 动力学参数

- **质量参数**: 整体质量 + 各连杆质量分布
- **惯性张量**: 各关节周围的转动惯量
- **摩擦系数**: 关节摩擦和接地面摩擦
- **阻尼系数**: 各关节的速度相关阻尼

### 2.2 动作空间自由度

#### 动作输入 (Action Space)

$$\mathbf{a}_t \in \mathbb{R}^{n_a}, \quad n_a = 10$$

**动作定义**:
$$\tau_i(t) = K_p^i(\theta_i^{ref}(a_t^i) - \theta_i(t)) - K_d^i \dot{\theta}_i(t)$$

其中：
- $a_t^i \in [-1, 1]$: 第 $i$ 个关节的归一化动作指令
- $\theta_i^{ref}(a_t^i) = \theta_i^{default} + \Delta\theta_{scale} \cdot a_t^i$: 参考关节角
- $\Delta\theta_{scale} = 0.25$ rad: 动作缩放系数
- $K_p^i$, $K_d^i$: 各关节的位置和速度增益

**控制频率**:
- 策略输出频率: 50 Hz (0.02s)
- 物理仿真频率: 200 Hz (0.005s)
- **控制步长比** (Decimation): 4 (每次策略输出对应4个仿真步)

### 2.3 观测空间自由度

#### 2.3.1 观测维度分解

$$\mathbf{o}_t = [\mathbf{o}_{proprio}, \mathbf{o}_{priv}, \mathbf{o}_{hist}]$$

**总观测维度**: $n_o = 39 + 43 + 187 + 390 = 659$ 维

##### (1) 本体感受观测 ($n_{proprio} = 39$)

| 分量 | 维数 | 含义 | 单位 |
|-----|------|------|------|
| 角速度 ($\boldsymbol{\omega}$) | 3 | 基座的角速度 (x,y,z) | rad/s |
| 欧拉角 ($\phi, \theta, \psi$) | 3 | 基座的姿态 (Roll, Pitch, Yaw) | rad |
| 命令 ($v_x, v_y, \omega_z$) | 3 | 目标速度和角速度 | m/s, rad/s |
| 关节位置 | 10 | 所有关节的相对角度 | rad |
| 关节速度 | 10 | 所有关节的角速度 | rad/s |
| 前一时刻动作 | 10 | 上一步的控制命令 | - |

**计算方式**:
$$\mathbf{o}_{proprio,t} = \begin{bmatrix} \omega_x(t) \cdot s_{\omega} \\ \omega_y(t) \cdot s_{\omega} \\ \omega_z(t) \cdot s_{\omega} \\ (\theta_x - \theta_x^{default}) \cdot s_{\theta} \\ \vdots \\ a_{t-1} \end{bmatrix}$$

其中观测缩放系数为: $s_{\omega} = 0.1$, $s_{\theta} = 1.0$, $s_{\dot{\theta}} = 0.05$

##### (2) 特权信息 ($n_{priv} = 43$)

这些信息仅在训练时可用，部署时不使用：

| 分量 | 维数 | 含义 |
|-----|------|------|
| 线性速度 | 3 | 基座的实际线性速度 |
| 接触状态 | 2 | 两个足端的接触标志 |
| 动力学延迟 | 18 | 各关节的感应延迟参数 |
| 质量参数 | 12 | 连杆质量分布参数 |
| 摩擦系数 | 12 | 各关节的摩擦系数 |
| 电机强度 | 10 | 电机最大输出力矩扩展系数 |
| 位置增益 (Kp) | 10 | 各关节的位置增益缩放 |
| 速度增益 (Kd) | 10 | 各关节的速度增益缩放 |
| 高度信息 | 4 | 足端高度测量 |

**特权信息的作用**: 支持 **Domain Randomization (域随机化)** 和 **Sim2Real Transfer (仿真到现实迁移)**

##### (3) LIDAR 扫描 ($n_{scan} = 187$)

- **类型**: 前向二维激光扫描
- **分辨率**: 187 束光线
- **扫描范围**: ±90° 的水平视角
- **最大距离**: 5m

$$\mathbf{o}_{scan} = [d_1, d_2, ..., d_{187}] \quad \text{(距离值)}$$

##### (4) 历史观测 ($n_{hist} = 390$)

保留过去 10 个时刻的本体感受信息:

$$\mathbf{o}_{hist,t} = [\mathbf{o}_{proprio,t-9}, ..., \mathbf{o}_{proprio,t-1}]$$

**作用**: 
- 提供时序信息用于 RNN 处理
- 帮助模型理解运动趋势和速度变化

#### 2.3.2 观测噪声

训练时加入高斯噪声以提高鲁棒性:

$$\tilde{\mathbf{o}}_t = \mathbf{o}_t + \mathbf{n}_t, \quad \mathbf{n}_t \sim \mathcal{N}(0, \boldsymbol{\Sigma}_n)$$

| 观测分量 | 噪声标准差 |
|---------|----------|
| 关节位置 | 0.03 rad |
| 关节速度 | 0.075 rad/s |
| 角速度 | 0.2 rad/s |
| 基座高度 | 0.02 m |

---

## 3. 强化学习框架

### 3.1 MDP 定义

该系统建立在**马尔可夫决策过程 (MDP)** 基础上：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

| 元素 | 定义 | 维度 |
|-----|------|------|
| 状态空间 $\mathcal{S}$ | 观测 $\mathbf{o}_t$ | 659 维 |
| 动作空间 $\mathcal{A}$ | 关节力矩指令 $\mathbf{a}_t$ | 10 维 |
| 状态转移 $\mathcal{P}$ | 物理引擎更新 (IsaacGym) | 连续 |
| 奖励函数 $\mathcal{R}$ | $r_t = R(\mathbf{s}_t, \mathbf{a}_t)$ | 标量 |
| 折扣因子 $\gamma$ | 0.998 | - |

### 3.2 策略与价值函数

#### 3.2.1 策略网络 (Actor)

$$\pi_{\boldsymbol{\theta}}(\mathbf{a}_t \mid \mathbf{o}_t) = \mathcal{N}(\boldsymbol{\mu}(s_t), \boldsymbol{\Sigma}(s_t))$$

**网络结构**:
```
Input: o_t (659 维)
       ↓
Encoder: [扫描编码器] + [历史编码器] (前馈 + RNN)
       ↓
Actor Hidden: [512] → [256] → [128]
       ↓
Output: μ (10 维), log(σ) (10 维)
```

**激活函数**: ELU (Exponential Linear Unit)

**输出分布**:
$$\mu_i = \tanh(\text{network output}_i) \quad \text{(限制在 [-1, 1])}$$
$$\sigma_i = \exp(\text{log}\sigma_i) \quad \text{(始终正)}$$

#### 3.2.2 价值函数 (Critic)

**奖励价值函数**:
$$V_{\boldsymbol{\phi}}^R(\mathbf{o}_t) \approx \mathbb{E}\left[\sum_{k=t}^{\infty} \gamma^{k-t} r_k \mid \mathbf{o}_t\right]$$

**成本价值函数**:
$$V_{\boldsymbol{\phi}}^C(\mathbf{o}_t) \approx \mathbb{E}\left[\sum_{k=t}^{\infty} \gamma^{k-t} c_k \mid \mathbf{o}_t\right]$$

**Critic 网络结构**:
```
Input: o_t (659 维)
       ↓
Critic Hidden: [512] → [256] → [128]
       ↓
Output: V^R (1 维) + V^C (9 维)
```

### 3.3 奖励函数设计

总奖励为多个子目标的加权和：

$$r_t = \sum_{i} w_i r_i(t)$$

#### 3.3.1 主要奖励项

| 奖励项 | 权重 | 含义 | 数学表达式 |
|-------|------|------|----------|
| 追踪线速度 | 2.5 | 鼓励沿命令方向移动 | $-\lambda_v \lVert v_x - v_x^{cmd} \rVert_2$ |
| 追踪角速度 | 2.0 | 鼓励按命令旋转 | $-\lambda_\omega \lVert \omega_z - \omega_z^{cmd} \rVert_2$ |
| 基座高度 | 0.2 | 保持稳定高度 | $-k_h(h - h^{target})^2$ |
| 脚步悬空时间 | 3.0 | 鼓励合理摆腿 | $+\lambda_{air} t_{air}$ |
| 双腿站立 | 6.0 | 鼓励双足支撑相 | $+\lambda_{2leg} \mathbb{1}[\text{both feet down}]$ |
| 无跳跃 | 0.7 | 抑制垂直跳跃 | $-\lambda_{jump} v_z^2$ |
| 方向控制 | 1.5 | 维持身体姿态 | $-k_\psi(\psi - \psi^{cmd})^2$ |
| 平滑性 | -0.01 | 减少抖动 | $-\lambda_{smooth}\lVert a_t - a_{t-1} \rVert_2^2$ |
| 能耗 | -2e-5 | 最小化功率消耗 | $-\lambda_p \sum_i \lvert \tau_i \dot{\theta}_i \rvert$ |

#### 3.3.2 奖励特性

- **稀疏/密集**: 密集型 (每步都有反馈)
- **延迟**: 无延迟 (当步返回)
- **范围**: 典型范围 $r_t \in [-10, 30]$ per step

### 3.4 成本函数设计

约束强化学习中包含 **9 种成本约束**:

$$\mathcal{C} = \{c_1(t), c_2(t), ..., c_9(t)\}$$

| 成本ID | 成本名称 | 约束目标 | 限值 (d) | 单位 |
|--------|---------|---------|---------|------|
| 1 | `torque_limit` | 限制关节力矩 | 45 | N·m |
| 2 | `pos_limit` | 限制关节位置超限 | 0.9 | 标准化 |
| 3 | `dof_vel_limits` | 限制关节速度 | 2.4 | rad/s |
| 4 | `vel_smoothness` | 速度平滑度 | 1.0 | - |
| 5 | `acc_smoothness` | 加速度平滑度 | 0.5 | - |
| 6 | `collision` | 躯干碰撞 | 0 | 次数 |
| 7 | `feet_contact_forces` | 足端接触力 | 120 | N |
| 8 | `stumble` | 摔跤事件 | 0.1 | 事件 |
| 9 | `foot_regular` / `trot_contact` | 相序约束 (步态周期) | 1.0 | - |

#### 3.4.1 成本计算示例

**力矩限制成本**:
$$c_{torque}(t) = \sum_i \max(0, \lvert \tau_i(t) \rvert - \tau_{max,i})$$

**关节位置限制成本**:
$$c_{pos}(t) = \sum_i \max(0, \lvert \theta_i(t) \rvert - \theta_{limit,i})$$

**步态相位约束成本** (Trot 步态):
$$c_{trot}(t) = \text{检测关节序列是否违反特定接触相位}$$

---

## 4. 约束PPO算法 (NP3O)

### 4.1 算法概述

NP3O 是对标准 PPO 的扩展，集成了**拉格朗日乘子法**处理约束：

$$\mathcal{L} = r_t - \lambda^T \mathbf{c}_t + \eta \text{KL}(\pi_{old} \| \pi_{new})$$

其中：
- $r_t$: 奖励回报
- $\mathbf{c}_t = [c_1(t), ..., c_9(t)]^T$: 成本向量
- $\boldsymbol{\lambda}$: 拉格朗日乘子 (9 维)
- $\eta$: KL 散度权重 (自适应)

### 4.2 核心组件

#### 4.2.1 优势估计

**奖励优势**:
$$A_t^R = \sum_{l=0}^{T-t} \gamma^l r_{t+l} + \gamma^T V^R(s_T) - V^R(s_t)$$

**成本优势** (GAE):
$$A_t^C = \sum_{l=0}^{T-t} \gamma^l c_{t+l}^j + \gamma^T V^C_j(s_T) - V^C_j(s_t)$$

其中 $T$ 为轨迹长度，$V^R$, $V^C$ 分别为奖励和成本价值函数。

**λ-回报 (GAE)**:
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

GAE 参数: $\gamma = 0.998$, $\lambda = 0.95$

#### 4.2.2 代理损失

**标准 PPO 损失** (奖励):
$$L^{CLIP} = -\mathbb{E}_t \left[\min(r_t \hat{A}_t^R, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t^R)\right]$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为重要性比率。

**成本约束代理损失**:
$$L^{COST} = \mathbb{E}_t \left[\max(c_t^j \cdot r_t, c_t^j \cdot \text{clip}(r_t, 1-\epsilon, 1+\epsilon))\right]$$

**违反项计算**:
$$L^{VIOL} = \sum_j k_j \cdot \text{ReLU}(L^{COST}_j - d_j)$$

其中 $k_j$ 为动态调整的拉格朗日乘子，$d_j$ 为约束限值。

#### 4.2.3 价值损失

**剪切价值损失** (Clipped Value Loss):
$$L^{V} = \mathbb{E}_t \left[\max((V - G_t)^2, (V_{clip} - G_t)^2)\right]$$

$$V_{clip} = V_{old} + \text{clip}(V - V_{old}, -\epsilon, \epsilon)$$

其中 $G_t$ 为目标价值估计。

#### 4.2.4 完整损失函数

$$L^{total} = L^{CLIP} + c_{viol} L^{VIOL} + c_v L^{V,R} + c_{vc} L^{V,C} + c_e L^{entropy}$$

其中权重系数为：
- $c_{viol} = 1.0$: 成本违反权重
- $c_v = 1.0$: 奖励价值损失权重
- $c_{vc} = 0.1$: 成本价值损失权重
- $c_e = 0.01$: 熵权重

### 4.3 训练超参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| 学习率 | $1 \times 10^{-3}$ | 初始 Adam 学习率 |
| 梯度截断 | 1.0 | 梯度范数上界 |
| PPO 裁剪参数 | 0.2 | $\epsilon$ 值 |
| 学习轮数 | 5 | 每次更新的学习轮次 |
| Mini-batch 数 | 4 | 小批次数 (每轮) |
| 滚动长度 | 1024 步 | 每次更新的轨迹长度 |
| 环境数 | 1024 | 并行环境数 |
| 最大迭代 | 30000 | 总训练迭代次数 |
| 保存间隔 | 5000 | 检查点保存频率 |

### 4.4 自适应机制

#### 4.4.1 自适应学习率

根据 KL 散度自动调整学习率：

$$\eta_t = \begin{cases}
\eta_{t-1} / 1.5 & \text{if } \text{KL}_{mean} > 2 \times \text{KL}_{target} \\
\eta_{t-1} \times 1.5 & \text{if } \text{KL}_{mean} < 0.5 \times \text{KL}_{target}
\end{cases}$$

其中 $\text{KL}_{target} = 0.01$。

#### 4.4.2 动态拉格朗日乘子

$$k_j(t) = \min(1.0, k_j(t-1) \times 1.0004^i)$$

其中 $i$ 为学习迭代次数。

---

## 5. 域随机化与Sim2Real迁移

### 5.1 域随机化参数

训练时在以下方面引入随机性：

| 参数类型 | 随机范围 | 目的 |
|---------|---------|------|
| 摩擦系数 | $\mu \in [0.5, 1.5]$ | 地面特性多样化 |
| 质量 | $\pm 15\%$ | 不确定性鲁棒性 |
| 阻尼 | $\pm 20\%$ | 关节变异性 |
| 电机延迟 | 0-3 步 | 通信延迟 |
| 传感器噪声 | 见第2.3.2节 | 测量不确定性 |
| 接触参数 | 关节摩擦、弹性 | 接地多样化 |

### 5.2 时间延迟建模

策略需要处理真实系统中的延迟：

$$\theta_i(t-\Delta t) \rightarrow \text{Policy} \rightarrow a_t \rightarrow \tau_i(t+\Delta t)$$

训练时随机采样的延迟：$\Delta t \in [0, 3 \times dt] = [0, 15\text{ ms}]$

---

## 6. 网络架构细节

### 6.1 完整网络结构

```
【输入层】(659维)
    ↓
【扫描编码器】(LIDAR → 中间表示)
  [187] → [256] → [128] → [64]
    ↓
【观测历史编码器】
  历史 [390] → RNN/LSTM → [128]
    ↓
【融合】
  [64] + [128] + [39] → [256]
    ↓
【Actor主干网络】
  [256] → ELU → [512] → ELU → [256] → ELU → [128]
    ↓
【策略头】
  μ: [128] → [10]
  σ: [128] → [10]
    ↓
【Critic主干网络】(共享特征)
  [256] → ELU → [512] → ELU → [256] → ELU → [128]
    ↓
【价值头】
  V^R: [128] → [1]
  V^C: [128] → [9]
```

### 6.2 网络参数量

| 模块 | 参数数 | 百分比 |
|-----|--------|--------|
| 编码器 | ~50K | 15% |
| Actor | ~200K | 60% |
| Critic | ~100K | 25% |
| **总计** | **~350K** | 100% |

---

## 7. 训练流程

### 7.1 单个训练迭代步骤

```
FOR each training iteration i = 1, 2, ..., 30000:
    
    【数据收集阶段】
    FOR each environment e = 1, ..., 1024:
        FOR each step t = 1, ..., 1024:
            o_t ← 环境观测
            a_t ~ π(·|o_t)  // 策略采样
            (o_{t+1}, r_t, c_t) ← 环境步进
            存储轨迹 (o_t, a_t, r_t, c_t, V^R_t, V^C_t)
    
    【回报计算】
    计算所有样本的 GAE 优势估计
    计算目标价值 G_t = A_t + V_t
    
    【策略更新】
    FOR learning_epoch = 1, ..., 5:
        FOR mini_batch in shuffle([所有样本]):
            计算新策略的对数概率
            计算代理损失 L^CLIP
            计算成本损失 L^COST 和 L^VIOL
            计算价值损失 L^V
            总损失 L = L^CLIP + w_viol*L^VIOL + w_v*L^V
            
            反向传播
            梯度裁剪
            优化器步进 (Adam)
    
    【动态参数更新】
    更新拉格朗日乘子 k_j
    自适应调整学习率

END FOR
```

### 7.2 分布式并行训练

- **1024 个并行环境**: 在 NVIDIA GPU 上并行运行
- **批处理大小**: 1024 × 1024 = 1M 样本/迭代
- **单次迭代时间**: ~2-3 秒 (V100 GPU)
- **总训练时间**: ~16-24 小时 (30000 迭代)

---

## 8. 性能指标

### 8.1 训练监测指标

| 指标 | 类型 | 说明 |
|-----|------|------|
| 平均回报 | 奖励 | 每个 episode 的累积奖励 |
| 成本违反量 | 约束 | $\sum_j \text{ReLU}(c_j - d_j)$ |
| 策略熵 | 多样性 | $\mathbb{H}(\pi) = \mathbb{E}[\text{log}\pi]$ |
| KL 散度 | 稳定性 | 新旧策略的 KL 距离 |
| 价值损失 | 拟合度 | 价值函数的预测误差 |

### 8.2 行为指标

- **前进速度**: 0.3 m/s (中速步行)
- **最大转向速度**: 1.5 rad/s
- **步频**: 2 Hz (0.5s per cycle)
- **步长**: 0.15 m (典型)
- **支撑时间比**: 60-70% (双足支撑比例)

---

## 9. 域迁移策略

### 9.1 Sim2Real Transfer

当将策略部署到真实机器人时：

1. **移除特权信息**: 丢弃仅用于训练的特权观测
2. **神经网络蒸馏**: 使用 LIDAR 和本体感受构建运行时策略
3. **微调**: 在真实机器人上少量微调
4. **频率匹配**: 保证控制周期与训练一致

### 9.2 策略推理优化

- **模型量化**: 转换为 ONNX 或 TVM 格式用于边缘部署
- **推理加速**: 使用 TensorRT 或其他运行时加速
- **内存效率**: 删除不必要的中间层以减少内存

---

## 10. 关键设计选择的理论依据

### 10.1 为什么使用约束RL?

约束强化学习通过显式建模成本/约束，直接优化安全性：
$$\arg\max_\pi \mathbb{E}[R] \quad \text{s.t.} \quad \mathbb{E}[C_j] \leq d_j$$

相比传统方法：
- **直接**：无需手调多个权重平衡
- **可控**：硬性保证约束满足
- **可验证**：明确的约束限值

### 10.2 为什么使用多个环境?

1024 个并行环境的目的：
- **样本效率**: 每迭代 1M 样本 (标准单环只有 4K)
- **估计方差**: 更好的蒙特卡洛估计
- **探索多样化**: 不同环境同时探索

### 10.3 为什么进行域随机化?

直接从单一仿真学习到现实的失败率高达 80%。域随机化通过：
- **过拟合抑制**: 防止适应单一仿真参数
- **鲁棒性增强**: 学习参数不变的策略
- **零-shot transfer**: 无需真实数据即可部署

---

## 11. 改进与扩展方向

### 11.1 可能的改进

1. **更多约束类型**: 添加能耗约束、安全区域约束等
2. **多目标优化**: 使用 Pareto 前沿权衡多个目标
3. **层级控制**: 高层规划 + 低层反应式控制
4. **迁移学习**: 基于现有策略快速适应新任务

### 11.2 理论开放问题

- 约束满足的样本复杂度下界?
- 最优拉格朗日乘子的收敛性?
- 域随机化覆盖真实分布的充要条件?

---

## 参考文献与资源

### 学术论文

1. Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
2. Ray et al. (2020): "RLlib: Abstractions for Distributed Reinforcement Learning"
3. Rudin et al. (2021): "Learning to Walk: Sim-to-Real Transfer"
4. Gangapurwala et al. (2023): "Rl-based Locomotion Control with RaiSim"

### 技术框架

- **IsaacGym**: NVIDIA GPU 加速仿真器
- **PyTorch**: 神经网络框架
- **TensorBoard/Wandb**: 训练监控

---

## 附录 A: 数学符号速查表

| 符号 | 含义 |
|-----|------|
| $s_t$ | 时刻 $t$ 的状态 |
| $a_t$ | 时刻 $t$ 的动作 |
| $r_t$ | 时刻 $t$ 的奖励 |
| $c_t^j$ | 时刻 $t$ 的第 $j$ 个成本 |
| $V(\cdot)$ | 价值函数 |
| $\pi(\cdot\mid s)$ | 策略 (条件分布) |
| $A_t$ | 优势估计 |
| $\gamma$ | 折扣因子 |
| $\lambda$ | GAE 参数 |

---

**文档版本**: 1.0  
**最后更新**: 2025年11月14日  
**维护者**: OmniBotSeries-Tinker 项目团队

