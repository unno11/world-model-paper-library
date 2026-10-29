# Offline RL：只用固定数据学习决策策略

离线强化学习（Offline Reinforcement Learning，Offline RL）从一个固定数据集学习策略，不在训练期间持续与环境交互。它适合真实机器人交互昂贵、危险或历史数据丰富的场景。

难点是：数据集只覆盖有限状态和动作，策略一旦选择数据中很少见的动作，价值估计可能不可靠。这叫分布外动作（Out-of-Distribution Action，OOD action）问题。

## 0. 学习目标

学完以后能够：

- 区分 Offline RL、Online RL、行为克隆和监督学习；
- 读懂固定数据集中的 transition 和 trajectory；
- 理解 distribution shift 与 extrapolation error；
- 解释 conservative、behavior regularization 和 policy constraint 的直觉；
- 判断数据覆盖、评估和部署风险；
- 在机器人和 World Model 场景中排查 offline RL 失败原因。

## 1. 为什么需要 Offline RL？

在线强化学习需要反复与环境交互：

```text
策略
↓
真实机器人执行
↓
得到 reward
↓
更新策略
↓
重复
```

真实交互可能慢、贵、危险。Offline RL 改为使用历史数据：

```text
历史轨迹 D
↓
状态、动作、奖励、下一状态
↓
学习 value / policy / model
↓
离线验证
↓
谨慎部署
```

它不仅是“把 replay buffer 放大”，因为训练策略不能随意获得新数据纠正错误。

## 2. 核心概念

### 2.1 Dataset

固定数据可能来自专家、旧策略、人类遥操作、随机探索或混合来源。质量、覆盖和行为策略决定学习上限。

### 2.2 Behavior Policy

行为策略 β(a|s) 是生成数据的策略。Offline RL 学习的目标策略 π 可能不同于 β，差异过大时风险增加。

### 2.3 Distribution Shift

训练数据中的状态—动作分布与新策略执行分布不同。状态偏移和动作偏移都会造成价值估计失真。

### 2.4 Extrapolation Error

价值网络在没有数据支持的动作上进行函数外推，可能产生错误的高 Q 值；策略又会选择这些虚假的高值动作，形成恶性循环。

### 2.5 Conservative Learning

保守方法倾向于不高估数据未覆盖的动作，或让策略不要偏离行为数据太远。不同算法实现不同，当前先掌握风险控制直觉。

### 2.6 Behavior Regularization

加入约束，使 π 不要离 β 太远，例如动作似然、KL penalty 或行为克隆混合目标。

## 3. Offline RL 与行为克隆的关系

行为克隆直接拟合专家动作：

```text
(o_t, a_t) → policy
```

Offline RL 还利用 reward 和未来回报：

```text
(o_t, a_t, r_{t+1}, o_{t+1})
↓
value / model / policy
↓
可能选择比数据中行为更好的动作
```

BC 通常更稳定但只能模仿数据；Offline RL 可能改进策略，但更容易受 OOD 价值错误影响。

## 4. 从零读固定 transition 数据

```python
batch = dataset.sample(batch_size)

states = batch["obs"]
actions = batch["action"]
rewards = batch["reward"]
next_states = batch["next_obs"]
dones = batch["done"]

q = critic(states, actions)
with torch.no_grad():
    next_action = target_policy(next_states)
    target_q = rewards + gamma * (1 - dones) * target_critic(
        next_states, next_action
    )
loss = mse_loss(q, target_q)
```

必须确认：

- batch 字段的 shape 和 dtype；
- done 是否包含截断；
- next_obs 是否真的是 action 后的状态；
- target action 是否超出数据分布；
- 数据是否按轨迹和时间正确加载。

## 5. 数据覆盖与质量

离线数据要问：

- 状态覆盖了哪些场景和任务阶段？
- 动作是否包含成功、失败和恢复行为？
- 是否有多种策略质量？
- 传感器是否同步？
- reward 是否可靠？
- train/validation 是否按 trajectory、实体和环境隔离？
- 数据中是否含大量重复帧？

数据多不等于数据有用，覆盖和质量更重要。

## 6. 评估与安全

离线评估可能包括：

- 数据集内行为预测；
- OPE（Off-Policy Evaluation）；
- value 分析；
- model-based rollout；
- 仿真；
- 受限真实部署。

离线指标不能完全替代真实闭环。机器人部署必须设置动作限制、回退策略、监控和急停。

## 7. 在具身智能和 World Model 中

Offline RL 数据常来自：

```text
人类遥操作
旧 policy
脚本控制器
随机/探索轨迹
失败与成功日志
↓
清洗、同步、切分
↓
behavior cloning / value / model-based learning
↓
仿真验证
↓
有限真实部署
```

World Model 可从固定数据学习 dynamics，再在模型中规划；但 model rollout 也可能进入数据未覆盖的区域。

## 8. 常见错误与 Debug

### 8.1 把数据集随机分样本

相邻 trajectory/frame 泄漏会导致验证虚高。按任务、场景、物体、机器人或时间切分。

### 8.2 只收成功轨迹

策略可能没有恢复失败、避障和边界行为，部署遇到偏差时无法纠正。数据质量应与目标风险相匹配。

### 8.3 Q 值异常高

检查 OOD action、target、reward scale、done mask、critic ensemble 和行为约束。

### 8.4 reward 标签错误

离线日志中 reward 可能由错误时间、成功标志或终止条件生成。手工审查少量轨迹。

### 8.5 OPE 过度可信

OPE 依赖假设，不能保证真实机器人安全。结合多个评估方式和受限部署。

## 9. 容易混淆

| 概念 | 含义 | 不等于 |
|---|---|---|
| Offline RL | 固定数据中学习决策 | 完全不需要评估 |
| Online RL | 可持续与环境交互 | 一定更安全 |
| BC | 拟合行为动作 | 利用长期 reward 改进 |
| behavior policy | 生成数据的策略 | 学习目标策略 |
| OOD action | 数据少/未覆盖动作 | 必然错误动作 |
| OPE | 无直接部署的策略评估 | 真实成功率 |
| conservative | 限制过度乐观外推 | 不学习新行为 |

## 10. 与其他笔记的关系

前置：[[MDP]]、[[Value_Function]]、[[Q_Learning]]、[[Imitation_Learning]]、[[模型评估]]、[[PyTorch]]。

相关：[[Actor_Critic]]、[[Model_Based_RL]]、[[视觉表征]]、[[损失函数]]、[[优化器]]。

本篇不展开具体 CQL、IQL、TD3+BC 等算法推导；当前先掌握离线数据和 OOD 风险。

## 11. 知识地图

```text
固定数据 D
↓
清洗、同步、按轨迹切分
↓
BC / value / model learning
↓
OOD 与 extrapolation risk
↓
OPE / simulation
↓
受限真实部署
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| data source | 专家、旧 policy、随机还是混合 |
| coverage | 场景、动作、成功/失败 |
| split | trajectory/entity/environment |
| transition | obs/action/reward/next_obs 对齐 |
| OOD | 策略是否偏离行为数据 |
| critic | Q 是否过高 |
| safety | 限幅、回退、急停 |

## 13. 练习题（题目下附参考答案）

1. Offline RL 的定义是什么？

**参考答案：** 训练期间只使用固定数据集学习策略，不持续与环境交互采集新数据。

2. behavior policy 是什么？

**参考答案：** 生成离线数据的原始策略。

3. OOD action 是什么？

**参考答案：** 数据集中很少出现或没有覆盖的动作。

4. extrapolation error 为什么危险？

**参考答案：** 价值网络可能在未见动作上错误高估，策略会主动选择这些虚假高值动作。

5. Offline RL 与行为克隆的区别是什么？

**参考答案：** BC 直接拟合动作，Offline RL 还利用 reward 和长期价值，可能改进但风险更高。

6. 为什么数据量多不等于覆盖好？

**参考答案：** 大量重复轨迹可能只覆盖少数场景和动作，不能支持策略泛化。

7. 为什么按 trajectory 切分？

**参考答案：** 防止同一连续轨迹的相邻样本同时进入训练和验证。

8. 为什么数据中要包含失败轨迹？

**参考答案：** 学习如何恢复、避障和识别风险；只看成功可能缺乏纠错行为。

9. conservative learning 的直觉是什么？

**参考答案：** 对数据未支持的动作保持保守，减少价值过度乐观外推。

10. behavior regularization 做什么？

**参考答案：** 约束学习策略不要过度偏离生成数据的 behavior policy。

11. OPE 能完全替代真实部署吗？

**参考答案：** 不能。OPE 依赖假设，真实闭环和安全条件仍需验证。

12. Q 值很高但策略很差怎么查？

**参考答案：** 检查 OOD action、target、reward、done、数据覆盖和 critic 外推。

13. next_obs 错位会怎样？

**参考答案：** value/dynamics 学到错误 transition，策略价值判断会偏离。

14. 离线数据的 reward 为什么需要人工抽查？

**参考答案：** 日志生成可能有时间、终止和成功标志错误，批量训练会放大错误。

15. World Model 如何帮助 Offline RL？

**参考答案：** 从固定数据学习 dynamics，在模型中进行短期规划或生成辅助经验，但必须控制模型漂移。

16. 仿真评估高能代表真实机器人安全吗？

**参考答案：** 不能，仍存在 sim-to-real、传感器、延迟和动力学差异。

17. 如何监控策略是否偏离行为数据？

**参考答案：** 比较动作似然、动作距离、状态分布和 rollout 中的 OOD 指标。

18. 为什么要限制真实部署动作？

**参考答案：** 离线策略可能在未覆盖状态选择危险动作，需要限幅、回退和急停。

19. Offline RL 中视觉 encoder 可能有什么问题？

**参考答案：** 视觉分布偏移、场景泄漏、latent 不保留动作信息或策略选到未见视觉状态。

20. 读 Offline RL 代码最先看什么？

**参考答案：** 数据来源/覆盖、transition 对齐、行为策略、OOD 约束、critic target 和评估安全流程。

## 13. 掌握度检查（附答案）

1. Offline RL 与 Online RL 的根本区别是什么？

**答案：** Offline RL 在固定数据中训练，Online RL 可持续与环境交互采样。

2. behavior policy 与 target policy 的区别是什么？

**答案：** 前者生成数据，后者是希望学习和部署的策略。

3. OOD action 为什么是核心风险？

**答案：** 价值函数在数据未覆盖动作上的估计缺乏可靠监督，可能错误高估。

4. Offline RL 为什么不能只当普通监督学习？

**答案：** 决策目标依赖未来奖励和动作后果，不只是复现静态标签。

5. BC 的优点是什么？

**答案：** 简单稳定，直接拟合示范，通常不需要价值外推。

6. BC 的局限是什么？

**答案：** 不能自动超越数据行为，且会受到状态分布偏移和误差累积影响。

7. conservative 方法解决什么直觉问题？

**答案：** 防止模型对数据外动作过度乐观。

8. 为什么数据切分要按 trajectory/entity？

**答案：** 保持验证独立，避免同一实体或相邻帧泄漏。

9. 为什么需要失败数据？

**答案：** 提供边界、恢复和风险行为的学习信号。

10. OPE 的限制是什么？

**答案：** 依赖数据覆盖和估计假设，不能完全保证真实闭环表现。

11. 如何判断 Q 值不可信？

**答案：** Q 异常高、与行为数据距离大、不同 critic 分歧大或真实回报不匹配。

12. Offline RL 部署前需要哪些保护？

**答案：** 动作限制、状态监控、回退策略、仿真测试和急停。

13. World Model 为什么也有 OOD 问题？

**答案：** rollout 可能离开固定数据分布，模型在这些状态上的预测不可靠。

14. 何时查 [[Imitation_Learning]]？

**答案：** 需要理解示范、行为克隆和专家数据质量时。

15. 读 Offline RL research code 最先确认哪五项？

**答案：** 数据覆盖、transition/时间、OOD 约束、价值 target、OPE/仿真/真实安全评估。
