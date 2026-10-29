# Policy Gradient：直接优化策略本身

策略梯度（Policy Gradient）直接调整策略 πθ(a|s)，使高回报动作出现的概率增加。它特别适合连续动作、随机策略和高维动作分布，是理解 PPO、REINFORCE、TRPO 等方法的基础。

本文负责 log-prob、trajectory return、policy gradient、探索和方差；价值基线与 advantage 见 [[Value_Function]]、[[Actor_Critic]]。

## 0. 学习目标

学完以后能够：

- 解释为什么可以直接优化策略；
- 理解 log π(a|s)、return 和 policy gradient；
- 读懂随机策略采样和 log_prob；
- 区分 on-policy 与 off-policy；
- 理解 baseline/advantage 如何降低方差；
- 排查 log_prob、动作分布、梯度和时间对齐错误。

## 1. 为什么直接优化策略？

Q-Learning 需要对动作价值建模，连续动作时动作最大化困难。策略梯度直接回答：

“在当前状态，应该以多大概率选择哪些动作？”

```text
state
↓
policy network
↓
action distribution πθ(a|s)
↓
sample action
↓
environment reward
↓
return / advantage
↓
提高高回报动作的 log probability
```

策略梯度通常是随机估计，方差较大，但能自然处理随机和连续动作。

## 2. 核心概念

### 2.1 Stochastic Policy

离散策略输出 logits/probabilities；连续策略常输出 Gaussian 的 mean 和 log_std，再采样动作。

### 2.2 Log Probability

采样动作 a 后，需要计算 log πθ(a|s)。策略目标常写成：

$$
L(\theta)=E[\log \pi_\theta(a_t|s_t)G_t]
$$

最大化它等价于提高高回报动作的概率；代码中常最小化负号后的 loss。

### 2.3 REINFORCE

REINFORCE 使用完整 episode return：

$$
\nabla_\theta J(\theta)=E[\nabla_\theta \log\pi_\theta(a_t|s_t)G_t]
$$

回报高时增加该动作概率，回报低时减少。

### 2.4 Baseline 与 Advantage

使用 baseline b(s) 不改变期望梯度，但能降低方差：

$$
A_t=G_t-b(s_t)
$$

常见 baseline 是 V(s)，于是得到 advantage。详细价值估计见 [[Value_Function]]。

## 3. 从零读策略采样

```python
dist = torch.distributions.Categorical(logits=policy_logits)
action = dist.sample()
log_prob = dist.log_prob(action)
entropy = dist.entropy()
```

- logits 不必先手动 Softmax；
- sample 产生探索动作；
- log_prob 是当前策略对该动作的对数概率；
- entropy 衡量分布不确定性，可作为探索正则。

连续高斯策略的 log_prob 还要注意 tanh squash 后的概率修正，初学阶段先确认库实现。

## 4. REINFORCE 流程

```text
采集一整条 trajectory
↓
计算每一步 return G_t
↓
计算 log_prob(a_t|s_t)
↓
policy_loss = - mean(log_prob × return)
↓
backward
↓
optimizer.step
```

必须使用当前策略重新计算或保存正确的 log_prob，且 trajectory 的 action、reward、state 要时间对齐。

## 5. On-Policy 与 Off-Policy

- on-policy：数据由当前策略采集，策略更新后旧数据通常失效；
- off-policy：可利用其他策略或旧数据，如 Q-Learning；
- policy gradient 基础方法通常是 on-policy；
- on-policy 数据效率低，但目标更直接。

## 6. Entropy 与探索

熵奖励鼓励策略保持一定随机性：

$$
L = L_{policy} - \beta H(\pi)
$$

beta 太大，策略一直随机；太小，可能过早变确定性。评估时通常不再按训练探索策略采样，需明确 deterministic 设置。

## 7. 在具身智能中的使用

策略梯度可用于：

- 连续关节动作；
- 末端位姿增量；
- 视觉策略；
- 高层技能选择；
- 随机探索和风险敏感控制。

真实机器人中每次交互昂贵，安全约束、奖励稀疏和样本效率是主要难点。模仿学习或 Model-Based 方法可减少纯在线探索需求，见 [[Imitation_Learning]]、[[Model_Based_RL]]。

## 8. 常见错误与 Debug

### 8.1 用概率而不是 log_prob

梯度公式使用 log probability。直接乘概率会改变目标和梯度尺度。

### 8.2 action 与 log_prob 错位

如果 batch 或时间索引错位，策略会把某动作归因于错误 reward。打印一个 trajectory 的 state/action/reward/log_prob。

### 8.3 return 未归一化导致方差大

可在 batch 内标准化 return/advantage，但需记录处理方式，避免训练和评估混淆。

### 8.4 entropy 系数过大

策略熵高、动作接近随机、成功率不升。查看 entropy 曲线和 beta。

### 8.5 旧数据用于当前策略更新

基础 on-policy 方法不应长时间复用旧 trajectory，除非算法有重要性采样或其他修正。

## 9. 容易混淆

| 概念 | 含义 | 不等于 |
|---|---|---|
| policy | 选择动作的分布/规则 | value |
| log_prob | 对动作概率取 log | loss 必然为正 |
| return | 未来回报样本 | 概率 |
| advantage | 相对 baseline 的回报 | Q 本身 |
| entropy | 分布不确定性 | 任务成功率 |
| on-policy | 当前策略采集数据 | 数据永远不能缓存 |
| baseline | 降方差参考 | 改变最优策略 |

## 10. 与其他笔记的关系

前置：[[MDP]]、[[Value_Function]]、[[概率统计]]、[[损失函数]]、[[优化器]]、[[PyTorch]]。

相关：[[Actor_Critic]]、[[Imitation_Learning]]、[[Model_Based_RL]]、[[Offline_RL]]。

本篇不展开 PPO clip、TRPO 约束等高级算法，只建立策略梯度的基本链路。

## 11. 知识地图

```text
policy network
↓
distribution
↓
sample action + log_prob
↓
trajectory return / advantage
↓
policy loss + entropy
↓
gradient update
↓
新策略采样
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| distribution | 离散/连续、参数含义 |
| action | sample 是否与 env 兼容 |
| log_prob | 与 action 一一对应 |
| return | 时间方向与 gamma |
| advantage | baseline、是否标准化 |
| entropy | 探索强度 |
| data | 当前策略采集还是旧数据 |

## 13. 练习题（题目下附参考答案）

1. 策略梯度直接优化什么？

**参考答案：** 直接优化策略参数，使高回报动作的概率增加。

2. log_prob 表示什么？

**参考答案：** 当前策略对已采样动作的对数概率。

3. 为什么最大化 log_prob×return？

**参考答案：** 高回报动作的概率被提高，低回报动作的概率相对降低。

4. REINFORCE 使用什么回报？

**参考答案：** 通常使用完整 episode 的 return。

5. baseline 的作用是什么？

**参考答案：** 降低策略梯度估计的方差，理想情况下不改变期望梯度。

6. advantage 常如何定义？

**参考答案：** return 或 Q 减去状态价值 V，表示动作相对基准好多少。

7. entropy 过高意味着什么？

**参考答案：** 策略分布不确定性大，动作更随机。

8. on-policy 是什么？

**参考答案：** 数据由当前策略采集并用于更新当前策略。

9. 为什么不能直接使用概率替代 log_prob？

**参考答案：** 策略梯度推导和梯度尺度基于 log probability，直接使用概率会改变目标。

10. 策略梯度如何处理连续动作？

**参考答案：** 让策略输出连续分布参数，如高斯均值和方差，再采样动作。

11. return 时间错位会怎样？

**参考答案：** reward 会被归因给错误动作，策略学习方向错误。

12. return 方差很大怎么办？

**参考答案：** 使用 baseline/advantage、回报标准化、合理 gamma 和更稳定的采样。

13. entropy 系数太大怎样表现？

**参考答案：** 策略持续随机，entropy 高，成功率长期不升。

14. 为什么 on-policy 样本效率低？

**参考答案：** 策略更新后旧数据不再严格对应当前策略，不能无限重复使用。

15. 机器人连续控制为何常用策略梯度类方法？

**参考答案：** 连续动作可直接由分布输出，不需要枚举或离散化所有动作。

16. 如何判断 action 与 log_prob 是否对齐？

**参考答案：** 打印相同时间步的 state、action、log_prob 和 reward，检查 batch/time 索引。

17. 训练时随机、评估时确定性有什么差异？

**参考答案：** 训练需要探索，评估通常使用均值或最大概率动作以稳定比较。

18. policy loss 下降但回报不升，检查什么？

**参考答案：** reward、return、log_prob、动作缩放、环境接口、探索系数和评估实现。

19. 模仿学习如何帮助策略梯度？

**参考答案：** 提供较好的初始化，减少早期随机探索和稀疏奖励困难。

20. 读 policy gradient 代码最先看什么？

**参考答案：** 策略分布、动作采样、log_prob、return/advantage、entropy 和数据是否 on-policy。

## 13. 掌握度检查（附答案）

1. 策略梯度与 Q-Learning 的基本差异是什么？

**答案：** 策略梯度直接优化动作分布，Q-Learning 学习动作价值再选动作。

2. 为什么需要 log_prob？

**答案：** 策略梯度使用 log π 的梯度来调整采样动作概率。

3. return 在策略更新中做什么？

**答案：** 提供动作好坏的回报信号。

4. baseline 为什么可以降低方差？

**答案：** 它去掉状态相关的平均回报，只保留动作相对优势。

5. advantage 是什么？

**答案：** 动作价值相对于状态价值的差异。

6. entropy 在策略中有什么作用？

**答案：** 鼓励探索，防止策略过早变得完全确定。

7. on-policy 为什么不能随意复用旧数据？

**答案：** 旧数据来自不同策略，梯度估计不再对应当前策略。

8. 连续策略常输出什么？

**答案：** 连续分布的参数，如 Gaussian mean 和 log standard deviation。

9. 为什么要检查动作缩放？

**答案：** 网络输出范围和环境 action_space 可能不同，错误会导致执行失真。

10. return 标准化有什么作用？

**答案：** 缩小梯度尺度波动，改善训练稳定性，但需保持实验记录清晰。

11. 真实机器人中的策略梯度难点是什么？

**答案：** 交互成本、安全、稀疏奖励和样本效率。

12. 为什么评估不能保留很大的 epsilon/随机性？

**答案：** 结果会高方差，难以公平比较策略能力。

13. 何时查 [[Actor_Critic]]？

**答案：** 需要用 critic 估计 baseline/advantage 并与 actor 联合训练时。

14. 何时查 [[Value_Function]]？

**答案：** 需要理解 return、V、Q、advantage 和 TD target 时。

15. 阅读策略梯度代码最先确认哪五项？

**答案：** policy distribution、sample/log_prob、return、advantage/baseline、entropy 和数据新鲜度。
