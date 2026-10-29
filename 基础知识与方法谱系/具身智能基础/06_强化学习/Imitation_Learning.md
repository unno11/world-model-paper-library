# Imitation Learning：从示范轨迹学习行为

模仿学习（Imitation Learning，IL）让 agent 从专家示范中学习行为。它适合真实机器人交互昂贵、奖励难设计或已有遥操作数据的场景。

本文重点负责行为克隆、示范数据、分布偏移、DAgger 和 diffusion/sequence policy 的基本联系；监督学习基础见 [[监督学习]]，策略优化见 [[Policy_Gradient]]、[[Actor_Critic]]，视觉输入见 [[视觉表征]]。

## 0. 学习目标

学完以后能够：

- 区分行为克隆、逆强化学习和强化学习；
- 读懂 observation-action 示范数据；
- 理解行为克隆为何本质上是监督学习；
- 解释 covariate shift 和 compounding error；
- 了解 DAgger、轨迹切分和序列策略；
- 调试动作时间对齐、数据泄漏、动作尺度和闭环失败。

## 1. 为什么需要模仿学习？

纯强化学习需要大量探索和奖励设计，真实机器人上可能危险且昂贵。若有专家轨迹：

```text
专家观察 o_t
↓
专家动作 a_t
↓
数据集 (o_t, a_t)
↓
policy(o_t) 预测动作
↓
监督 loss
↓
部署并执行
```

行为克隆先把专家行为变成可学习的输入—目标对，但离线准确不保证闭环成功。

## 2. 核心概念

### 2.1 Demonstration

demonstration 是专家完成任务的轨迹，可能来自人类遥操作、脚本规划器、已有策略或真实日志。

### 2.2 Behavior Cloning

行为克隆（Behavior Cloning，BC）把动作当标签：

$$
\min_\theta E_{(o,a)\sim D}[L(\pi_\theta(o),a)]
$$

连续动作可用 MSE/Huber，离散动作可用交叉熵。它属于监督学习，但数据是序贯轨迹。

### 2.3 Covariate Shift

专家数据只覆盖专家访问的状态；策略产生轻微错误后进入新状态，而训练中没见过这些状态，错误可能继续扩大。

```text
小动作误差
↓
状态偏离专家轨迹
↓
看到未覆盖观测
↓
动作更不准
↓
误差累积
```

### 2.4 DAgger

DAgger 让当前策略执行，遇到的状态由专家重新标注动作，再把新数据加入训练集。它缓解训练/部署状态分布差异，但需要在线专家或安全标注。

### 2.5 Inverse Reinforcement Learning

逆强化学习试图从专家行为推断奖励或任务目标，再进行 RL。它与 BC 不同：BC 直接拟合动作，IRL 学习解释行为的奖励。

## 3. 从零读行为克隆代码

```python
for obs, expert_action in loader:
    pred_action = policy(obs)
    loss = torch.nn.functional.mse_loss(pred_action, expert_action)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

要检查：

- obs 是否包含正确时间上下文；
- expert_action 是否是当前 obs 后要执行的动作；
- action 单位和范围是否一致；
- 连续动作 loss 的 reduction；
- train/validation 是否按 trajectory/scene 切分。

## 4. 轨迹数据结构

一条轨迹可能包含：

```text
observations[t]
actions[t]
rewards[t]
dones[t]
timestamps[t]
camera frames[t]
robot states[t]
task metadata
```

不要随意把所有时间步随机打散后切分。相邻帧、同一物体和同一任务可能泄漏到验证集。

## 5. 序列策略

如果单个 observation 不足以决定动作，可以输入历史窗口：

```text
[o_{t-L+1}, ..., o_t]
↓
CNN/ViT frame encoder
↓
RNN/LSTM/Transformer
↓
policy head
↓
a_t 或 action chunk
```

action chunk 一次预测未来多个动作，可减少控制抖动，但要明确执行和重规划方式。

## 6. 在具身智能中的使用

模仿学习常用于：

- 视觉行为克隆；
- 机器人操作；
- 多任务技能学习；
- 语言条件动作；
- 轨迹初始化；
- RL policy warm start；
- World Model 的数据收集。

真实项目流程：

```text
读取示范日志
↓
同步相机、状态、动作
↓
清理异常轨迹
↓
构造 observation/action window
↓
训练 policy
↓
离线指标
↓
仿真/真实闭环
↓
失败轨迹再收集
```

## 7. 常见错误与 Debug

### 7.1 action 错位

o_t 对应 a_t 还是 a_{t+1}？打印时间戳和一条完整轨迹，确认动作执行时刻。

### 7.2 动作尺度错误

数据可能是关节位置、速度、增量或归一化 [-1,1]。预测正确形状不代表单位正确。

### 7.3 训练随机打散导致泄漏

按样本随机切分相邻帧会让验证过于容易。按轨迹、场景、物体或机器人划分。

### 7.4 训练 loss 低、闭环失败

原因可能是 covariate shift、观察延迟、动作执行误差、视觉分布偏移和错误累计。

### 7.5 专家数据质量不一致

慢动作、抖动、失败示范和不同控制风格会增加标签噪声。需要过滤、分层或条件化。

## 8. 容易混淆

| 概念 | 含义 | 区别 |
|---|---|---|
| BC | 直接拟合专家动作 | 不显式学习 reward |
| RL | 通过环境反馈优化 | 可超越专家但交互贵 |
| DAgger | 用当前策略状态再请专家标注 | 需要在线标注 |
| IRL | 从行为推断奖励 | 再进行决策优化 |
| covariate shift | 训练/部署状态分布变 | 不等于标签噪声 |
| action chunk | 一次预测多个动作 | 不等于 episode |
| offline metric | 离线预测误差 | 不等于闭环成功 |

## 9. 与其他笔记的关系

前置：[[MDP]]、[[监督学习]]、[[损失函数]]、[[模型评估]]、[[PyTorch]]。

相关：[[Policy_Gradient]]、[[Actor_Critic]]、[[Model_Based_RL]]、[[Offline_RL]]、[[视觉表征]]、[[Transformer]]。

本篇不展开 diffusion policy 的完整数学推导；当前先掌握示范数据和闭环问题。

## 10. 知识地图

```text
expert trajectories
↓
observation/action alignment
↓
behavior cloning
↓
offline validation
↓
closed-loop execution
├─ success
├─ distribution shift
└─ collect corrective data / DAgger
```

## 11. 快速查询表

| 检查项 | 重点 |
|---|---|
| observation | 当前还是历史窗口 |
| action | 时刻、单位、范围 |
| split | trajectory/scene/entity 隔离 |
| loss | 连续/离散目标匹配 |
| offline metric | 只说明预测，不代表控制 |
| rollout | 真实闭环是否评估 |
| failure | 是否回收新数据 |

## 12. 练习题（题目下附参考答案）

1. 模仿学习使用什么数据？

**参考答案：** 专家示范轨迹，包括观测、动作和可能的状态、奖励、时间戳。

2. 行为克隆本质上是什么？

**参考答案：** 把专家动作当标签的监督学习。

3. covariate shift 是什么？

**参考答案：** 训练时看到专家状态分布，部署时策略错误导致进入不同状态分布。

4. 为什么闭环误差会累积？

**参考答案：** 当前小动作误差改变下一状态，下一状态又可能超出训练覆盖，使后续预测更差。

5. DAgger 做什么？

**参考答案：** 让当前策略访问状态，再由专家给这些状态标注动作并加入训练。

6. BC 的连续动作常用什么 loss？

**参考答案：** MSE、MAE 或 Huber，需结合动作尺度和异常值选择。

7. o_t 与 a_t 对齐意味着什么？

**参考答案：** a_t 是在观测 o_t 时刻应执行的动作，具体定义要以控制接口为准。

8. 为什么不能随机切分相邻帧？

**参考答案：** 相邻帧高度相似，训练和验证会泄漏同一轨迹信息。

9. action chunk 有什么作用？

**参考答案：** 一次预测多个未来动作，可能减少抖动并提高规划效率。

10. 动作归一化错误会怎样？

**参考答案：** 即使 MSE 很低，实际执行单位/幅度也可能完全错误。

11. BC 的 offline loss 低但闭环失败可能为什么？

**参考答案：** 分布偏移、时间错位、延迟、执行误差或训练数据覆盖不足。

12. 如何检查 action 对齐？

**参考答案：** 打印时间戳、观测、动作、执行时间和下一观测，手工核对一条轨迹。

13. 专家示范为何需要清洗？

**参考答案：** 失败、抖动、延迟和风格不一致会成为噪声或错误目标。

14. BC 如何作为 RL 初始化？

**参考答案：** 先用示范训练一个初始策略，再用 RL 通过环境奖励继续改进。

15. DAgger 的成本是什么？

**参考答案：** 需要策略在线运行和专家实时标注，成本与安全要求较高。

16. 视觉 BC 中 encoder 输出什么？

**参考答案：** 图像 feature/latent，再与状态融合后生成动作。

17. 真实机器人部署前检查什么？

**参考答案：** 动作范围、频率、延迟、观测归一化、相机分布、急停和失败保护。

18. 为什么单个 observation 可能不够？

**参考答案：** 部分可观测或动作依赖历史，需要帧堆叠、RNN 或 Transformer。

19. offline accuracy 能代替 success rate 吗？

**参考答案：** 不能，动作预测正确不一定形成成功的闭环轨迹。

20. 读模仿学习代码最先看什么？

**参考答案：** 数据字段、时间对齐、窗口构造、action scaling、split、loss 和闭环评估。

## 13. 掌握度检查（附答案）

1. 模仿学习和强化学习的主要差别是什么？

**答案：** 模仿学习利用专家示范，强化学习主要利用环境奖励和交互。

2. 行为克隆为什么属于监督学习？

**答案：** observation 是输入，专家 action 是标签，直接最小化预测误差。

3. covariate shift 如何产生？

**答案：** 策略部署时进入专家数据未覆盖的状态，输入分布发生改变。

4. 为什么误差会累积？

**答案：** 当前错误改变未来状态，未来错误又建立在偏离状态上。

5. DAgger 如何缓解分布偏移？

**答案：** 收集当前策略实际访问状态的专家标签，把这些状态加入训练数据。

6. 为什么轨迹切分比样本切分重要？

**答案：** 样本有时间和实体相关性，随机样本切分会泄漏相邻信息。

7. action chunk 的好处和风险是什么？

**答案：** 好处是平滑和效率，风险是错误会持续多个动作且需要正确执行/重规划。

8. 为什么动作尺度必须记录？

**答案：** 网络值、归一化值和物理单位不同，错误缩放会直接导致控制失败。

9. offline 指标说明什么？

**答案：** 说明对静态数据的拟合，不自动说明闭环策略表现。

10. 专家数据质量如何影响 BC？

**答案：** 噪声、失败和风格冲突会使策略学习不一致行为。

11. 视觉 BC 需要保留哪些视觉信息？

**答案：** 目标身份、位置、姿态、深度和动作相关时序信息。

12. 为什么可能需要历史窗口？

**答案：** 当前观测不完整，历史能提供速度、阶段和隐藏状态线索。

13. 何时查 [[MDP]]？

**答案：** 需要确认 observation、action、transition、reward 和时间关系时。

14. 何时查 [[Offline_RL]]？

**答案：** 需要从固定数据利用奖励学习并处理分布外动作时。

15. 阅读行为克隆研究代码最先确认哪五项？

**答案：** 数据来源、时间对齐、action scaling、切分方式、离线与闭环评估。
