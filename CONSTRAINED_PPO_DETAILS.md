# 约束PPO算法数学推导与实现细节

## 1. 强化学习基础回顾

### 1.1 策略梯度定理

**定理 (Policy Gradient Theorem)**:
$$\nabla J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]$$

其中：
- $J(\theta) = \mathbb{E}[R_0]$: 目标函数 (初始状态下的期望回报)
- $d^\pi(s)$: 策略 $\pi$ 下的状态分布
- $Q^\pi(s,a)$: 动作价值函数 = $\mathbb{E}[G_t | s_t = s, a_t = a]$

### 1.2 优势函数

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**意义**：相对于平均价值的超额收益。

### 1.3 高斯策略的参数化

对于连续控制，使用高斯策略：
$$\pi_\theta(a|s) = \mathcal{N}(a | \mu_\theta(s), \sigma^2_\theta(s))$$

**对数似然梯度**:
$$\nabla_\theta \log \pi_\theta(a|s) = \frac{1}{\sigma^2(s)}\nabla_\theta \mu(s) \cdot (a - \mu(s)) + \nabla_\theta \sigma(s) \cdot \frac{(a-\mu(s))^2 - \sigma^2(s)}{\sigma^3(s)}$$

---

## 2. PPO 算法完整推导

### 2.1 重要性采样与异策略修正

对于来自旧策略 $\pi_{old}$ 采样的数据，新策略 $\pi_{new}$ 的期望回报可表示为：

$$J(\pi_{new}) = \mathbb{E}_{s,a \sim \pi_{old}} \left[\frac{\pi_{new}(a|s)}{\pi_{old}(a|s)} Q^{\pi_{old}}(s,a)\right]$$

定义重要性比率：
$$r(a|s) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$$

### 2.2 PPO-Clip 目标

为防止策略变化过大，PPO 引入裁剪：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)\right]$$

其中 $\epsilon = 0.2$ 是裁剪范围。

**几何直观**:
- 如果优势 $\hat{A}_t > 0$ (有利的动作)：
  - 不裁剪时，$r_t$ 越大损失越小 (鼓励增大概率)
  - 裁剪时，$r_t \in [1-\epsilon, 1+\epsilon]$ 时停止增长
  
- 如果优势 $\hat{A}_t < 0$ (不利的动作)：
  - 类似地，$r_t$ 在 $[1-\epsilon, 1+\epsilon]$ 之间变化

### 2.3 Generalized Advantage Estimation (GAE)

**目标回报**:
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...$$

**TD 残差** (Temporal Difference):
$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**GAE Lambda 回报**:
$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

**性质**:
- $\lambda = 0$: 仅用一步 TD 误差
- $\lambda = 1$: 蒙特卡洛回报
- $\lambda = 0.95$: 平衡偏差和方差

**计算递推关系**:
$$\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$$

向后计算可在 $O(T)$ 时间内完成。

### 2.4 价值函数损失

**无裁剪**:
$$L^V(\phi) = \frac{1}{2} \mathbb{E}_{t} \left[(V_\phi(s_t) - G_t)^2\right]$$

**有裁剪** (可选，PPO 中推荐):
$$L^{V,clip}(\phi) = \mathbb{E}_t \left[\max((V_\phi - G_t)^2, (V_{clip} - G_t)^2)\right]$$

其中:
$$V_{clip} = V_{\phi,old} + \text{clip}(V_\phi - V_{\phi,old}, -\epsilon, \epsilon)$$

---

## 3. 约束PPO (约束强化学习)

### 3.1 约束MDP 问题定义

目标：
$$\max_\pi \mathbb{E}[R] = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

约束条件：
$$\mathbb{E}[C_j] = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t c_j(t)\right] \leq d_j, \quad j = 1, ..., m$$

其中 $d_j$ 是第 $j$ 个约束的限值。

### 3.2 Lagrangian 对偶形式

引入拉格朗日乘子 $\lambda_j \geq 0$，构造拉格朗日函数：

$$L(\pi, \lambda) = \mathbb{E}[R] - \sum_{j=1}^m \lambda_j (\mathbb{E}[C_j] - d_j)$$

**鞍点问题**:
$$\pi^* = \arg\max_\pi \min_\lambda L(\pi, \lambda)$$

### 3.3 Policy Gradient 推导

对策略参数求导：
$$\nabla_\theta L = \nabla_\theta \mathbb{E}[R] - \sum_j \lambda_j \nabla_\theta \mathbb{E}[C_j]$$

$$= \mathbb{E}_{s,a \sim \pi} [\nabla_\theta \log \pi_\theta(a|s) (Q^R - \sum_j \lambda_j Q^C_j)]$$

**实际实现中的 modified advantage**:
$$\tilde{A}_t = A_t^R - \sum_j \lambda_j A_t^C_j$$

### 3.4 代理损失组件

#### 奖励代理损失
$$L^{CLIP,R} = \hat{\mathbb{E}}_t \left[\min(r_t \hat{A}_t^R, \text{clip}(r_t) \hat{A}_t^R)\right]$$

#### 成本代理损失
对于每个约束 $j$：
$$L^{CLIP,C}_j = \hat{\mathbb{E}}_t \left[\max(r_t \hat{A}_t^C_j, \text{clip}(r_t) \hat{A}_t^C_j)\right]$$

**注意**：成本代理不取最小值，因为我们希望 *最大化* 成本 (推高成本以满足约束)。

#### 约束违反项
$$L^{VIOL}_j = \hat{\mathbb{E}}_t [G_t^C_j] - d_j$$

其中 $G_t^C_j$ 是成本的目标回报。

$$L^{VIOL} = \sum_j k_j \cdot \text{ReLU}(L^{VIOL}_j)$$

### 3.5 Lagrangian 乘子更新

**随机梯度上升**:
$$\lambda_j^{(i+1)} = \lambda_j^{(i)} + \alpha \cdot L^{VIOL}_j$$

**实际中采用的递推关系**:
$$k_j^{(i)} = \min(1.0, k_j^{(i-1)} \cdot (1.0004)^i)$$

这相当于每个迭代增加 0.04% 的 Lagrangian 乘子系数。

---

## 4. 在 Tinker 项目中的具体实现

### 4.1 完整损失函数

在每个学习轮次中：

```python
# 1. 前向传播获取新策略输出
log_prob_new = actor_critic.get_actions_log_prob(actions_batch)
values_r = actor_critic.evaluate(critic_obs, ...)  # 奖励价值
values_c = actor_critic.evaluate_cost(critic_obs, ...)  # 成本价值

# 2. 计算 surrogate 损失 (奖励)
ratio = torch.exp(log_prob_new - log_prob_old)
l_clip = torch.clamp(ratio, 1-clip, 1+clip)
surrogate = torch.min(ratio * adv_r, l_clip * adv_r)
loss_r = -surrogate.mean()

# 3. 计算成本 surrogate
surrogate_c = cost_adv * ratio  # (batch, n_costs)
surrogate_c_clip = cost_adv * torch.clamp(ratio, 1-clip, 1+clip)
surrogate_c_max = torch.max(surrogate_c, surrogate_c_clip)  # (batch, n_costs)

# 4. 计算约束违反
cost_violation = G_c - d  # (batch, n_costs)
loss_c = self.k_value * F.relu(surrogate_c_max.mean(0) + cost_violation.mean(0))
loss_c = loss_c.sum()  # 对所有成本求和

# 5. 价值损失
v_clipped = v_old + torch.clamp(v - v_old, -clip, clip)
loss_v_r = torch.max((v - G_r)**2, (v_clipped - G_r)**2).mean()
loss_v_c = torch.max((v_c - G_c)**2, (v_c_clipped - G_c)**2).mean()

# 6. 熵正则
entropy = actor_critic.entropy.mean()
loss_entropy = -entropy_coef * entropy

# 7. 总损失
loss_total = loss_r + cost_viol_coef * loss_c + \
             value_loss_coef * loss_v_r + cost_value_loss_coef * loss_v_c + \
             loss_entropy
```

### 4.2 单个成本函数实现示例

**力矩限制成本** (`_cost_torque_limit`):
```python
def _cost_torque_limit(self):
    tau_exceeded = torch.clamp(torch.abs(self.torques) - 45, min=0)
    return tau_exceeded.sum(dim=1)  # (num_envs,)
```

**步态相位约束** (`_cost_trot_contact`):
```python
def _cost_trot_contact(self):
    # 对于 Trot 步态，应遵循特定的接触相序
    # 例如：左前-右后 → 双足 → 左后-右前 → 双足
    
    left_contact = self.contact_filt[:, 0]
    right_contact = self.contact_filt[:, 1]
    
    # 检查是否违反预期的相位
    violation = torch.abs(
        (left_contact - right_contact).float() - 
        expected_phase[self.global_step % cycle_length]
    )
    return violation
```

### 4.3 观测与价值函数映射

```python
class ActorCriticRMA(nn.Module):
    def __init__(self, n_proprio, n_scan, n_obs, n_priv_latent, 
                 history_len, n_actions, **cfg):
        # 输入维度 = 本体感受 + 扫描 + 特权信息 + 历史
        self.obs_encoder = ScanEncoder(n_scan, scan_hidden_dims)
        self.history_encoder = HistoryEncoder(n_proprio, history_len, ...)
        
        # 政策网络 (共享 + 头部)
        self.actor_backbone = MLP(encoded_dims, actor_hidden_dims)
        self.mu_head = nn.Linear(actor_hidden_dims[-1], n_actions)
        self.log_sigma = nn.Parameter(torch.zeros(n_actions))
        
        # 价值网络 (共享 + 头部)
        self.critic_backbone = MLP(encoded_dims, critic_hidden_dims)
        self.v_r_head = nn.Linear(critic_hidden_dims[-1], 1)      # 奖励
        self.v_c_head = nn.Linear(critic_hidden_dims[-1], 9)      # 9个成本
    
    def act(self, obs):
        encoded = self.encode_obs(obs)
        feat = self.actor_backbone(encoded)
        mu = self.mu_head(feat)
        return mu
    
    def evaluate(self, obs):
        # 返回奖励价值
        encoded = self.encode_obs(obs)
        feat = self.critic_backbone(encoded)
        return self.v_r_head(feat)
    
    def evaluate_cost(self, obs):
        # 返回9维成本价值
        encoded = self.encode_obs(obs)
        feat = self.critic_backbone(encoded)
        return self.v_c_head(feat)  # (batch, 9)
```

---

## 5. 收敛性分析

### 5.1 理论收敛保证

**定理 (PPO 收敛)**:
在以下条件下，PPO 以概率 1 收敛到局部最优：
1. 策略参数化是紧凑的
2. 学习率满足 $\sum_i \alpha_i = \infty$, $\sum_i \alpha_i^2 < \infty$
3. 折扣因子 $\gamma$ 足够小

### 5.2 约束满足的收敛速度

对于约束 PPO，约束违反的收敛速度为 $O(1/\sqrt{N})$，其中 $N$ 是样本数。

**证明思路**:
1. 每个 PPO 更新使奖励改进
2. 拉格朗日乘子更新逐步强化约束
3. 最终收敛到约束可行域附近

### 5.3 实践中的收敛迹象

观察以下指标：
- 回报曲线变平缓 (plateau)
- 成本违反趋于限值 $d_j$
- 策略熵稳定在某个值
- KL 散度在目标范围内

---

## 6. 常见问题与调试

### 6.1 为什么成本约束没有被满足?

**可能原因**:
1. **拉格朗日乘子初始化过小**: $k_0$ 应该大于 0
2. **成本目标过严格**: 检查 $d_j$ 值是否可达
3. **学习率过高**: 导致策略震荡

**解决方案**:
```python
# 增加约束权重
self.k_value *= 2.0

# 或者放松约束目标
self.d_values[j] *= 1.5
```

### 6.2 为什么奖励下降而成本满足?

**可能原因**:
- 拉格朗日乘子过大，过度惩罚成本

**解决方案**:
```python
# 检查成本与奖励的权衡
# 尝试调整 cost_viol_loss_coef
cost_viol_loss_coef = 0.5  # 从 1.0 降低
```

### 6.3 为什么策略探索不足?

**可能原因**:
- 学习率太高，收敛过快
- 成本约束过紧，限制了探索空间

**解决方案**:
```python
# 增加熵系数
entropy_coef = 0.05  # 从 0.01 增加

# 或者在训练初期放松约束
if iteration < 5000:
    k_value *= 0.5
```

---

## 7. 性能优化技巧

### 7.1 计算效率

**向量化**: 所有操作在 PyTorch 张量上进行
```python
# 低效 (Python 循环)
for i in range(n_envs):
    cost[i] = compute_cost_i(state[i])

# 高效 (向量化)
cost = compute_cost_vectorized(state)  # (n_envs,)
```

### 7.2 内存效率

**轨迹存储**:
- 使用循环缓冲区避免重复分配
- 及时清理不需要的张量

### 7.3 梯度计算加速

- 使用 `torch.no_grad()` 防止计算不需要的梯度
- 对特征提取使用 `detach()` 断开梯度流

---

## 8. 参考论文的关键结果

### Schulman et al. (2017) - PPO

原始论文展示：
- PPO 在连续控制上超过 A3C 和 TRPO
- 简化的算法实现但性能不弱
- 在 Mujoco benchmark 上的 SOTA 成绩

### Achiam et al. (2017) - Constrained Policy Optimization

约束 RL 的理论基础：
- 拉格朗日乘子方法的收敛性证明
- 约束满足的充分条件
- 与无约束 RL 的关系分析

### Raghunathan et al. (2021) - Constrained RL for Robotics

应用于机器人的约束 RL：
- 实际机器人实验验证理论
- Sim2Real 迁移的约束保持方法
- 多约束的处理策略

---

## 附录 B: 实现检查清单

在调试约束 PPO 实现时：

- [ ] 成本函数正确计算 (单位、符号)
- [ ] GAE 回报计算无溢出
- [ ] 优势估计不包含 NaN
- [ ] 价值函数初始化合理
- [ ] 重要性比率在 $[0.5, 2.0]$ 范围内
- [ ] KL 散度在 $[0.01, 0.1]$ 范围内
- [ ] 拉格朗日乘子单调增加
- [ ] 成本约束逐步满足
- [ ] 奖励和成本没有明显相关
- [ ] GPU 内存使用稳定 (无泄漏)

---

**版本**: 2.0  
**最后更新**: 2025年11月14日

