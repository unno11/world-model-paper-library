# Model-Based RL：学习环境模型再进行规划

Model-Based Reinforcement Learning（基于模型的强化学习，Model-Based RL）先学习或使用环境的动态模型，再在模型中预测未来、规划动作或辅助训练策略。

它与 World Model、机器人规划和 model predictive control 紧密相关。MDP 基础见 [[MDP]]，价值与策略见 [[Value_Function]]、[[Actor_Critic]]，表征和 Transformer 见 [[视觉表征]]、[[Transformer]]。

## 0. 学习目标

学完以后能够：

- 区分 model-free RL 与 model-based RL；
- 解释 transition model、reward model 和 terminal model；
- 读懂 imagined rollout 和 MPC 的基本流程；
- 理解模型误差、长时预测漂移和 uncertainty；
- 将视觉 latent、动作和未来预测连接到 World Model；
- 调试时间错位、模型泄漏、rollout horizon 和规划失败。

## 1. 为什么需要环境模型？

真实机器人交互昂贵、慢且有风险。如果能预测动作后的未来：

```text
当前状态 s_t
↓
候选动作 a_t
↓
模型预测 s_{t+1}, r_{t+1}
↓
继续预测若干步
↓
比较候选轨迹
↓
执行最好的动作
```

模型可以减少真实试错，但错误预测也会让规划走向不存在的未来。

## 2. 核心概念

### 2.1 Transition Model

transition model 预测：

$$
\hat{s}_{t+1}=f_\phi(s_t,a_t)
$$

也可以预测分布 p(s_{t+1}|s_t,a_t)，表达不确定性。

### 2.2 Reward Model

预测动作后的奖励：

$$
\hat{r}_{t+1}=r_\phi(s_t,a_t)
$$

如果 reward 由环境直接计算，可能不需要单独学习；真实复杂任务中也可能学习成功/终止模型。

### 2.3 Observation Model

视觉系统中，模型可能预测下一 observation、latent 或 observation 的特征。直接像素预测难，常在 latent space 预测。

### 2.4 World Model

World Model 通常包含：

- encoder：观测 → latent；
- dynamics：latent + action → next latent；
- reward/termination head；
- decoder（可选）：latent → observation；
- planner/policy：利用模型预测未来。

```text
image o_t
↓ encoder
z_t
↓ dynamics(z_t,a_t)
z_{t+1}
↓ reward/terminal prediction
r_{t+1}, done
↓ planner
a_t
```

### 2.5 Imagined Rollout

在学习模型中展开未来轨迹，称为 imagined rollout。它不是真实环境交互，质量取决于模型在规划状态分布上的准确性。

### 2.6 Model Predictive Control

MPC 每次只执行规划序列的第一步，然后获得新观测、重新规划。这样可以减少长期预测误差影响。

## 3. Model-Based RL 的整体流程

```text
真实数据
↓
训练 dynamics/reward model
↓
用模型预测候选未来
↓
planner / policy 比较回报
↓
执行第一步动作
↓
获取真实新状态
↓
更新模型或重新规划
```

有些方法交替进行模型学习和策略学习；有些使用模型生成额外数据辅助 actor/critic。

## 4. 从零读模型预测代码

```python
z = encoder(obs)
next_z = dynamics(torch.cat([z, action], dim=-1))
pred_reward = reward_head(next_z)

target_z = target_encoder(next_obs).detach()
loss_state = mse_loss(next_z, target_z)
loss_reward = mse_loss(pred_reward, reward)
loss = loss_state + loss_reward
```

要检查：

- obs 与 next_obs 是否相邻且时间正确；
- action 是否是从 obs 执行得到的动作；
- target encoder 是否按设计 detach；
- latent shape 是否一致；
- reward head 的输入时刻是否正确。

## 5. 规划与 rollout horizon

规划 horizon 越长，能看得更远，但模型误差累积越严重，计算也更贵。horizon 太短可能只优化眼前奖励，太长可能相信错误未来。

常见策略：

- 短 horizon MPC；
- beam search 或采样候选动作；
- CEM 优化动作序列；
- actor 提议、critic 评价；
- uncertainty penalty；
- 定期用真实观测校正。

当前知道这些方法存在即可，重点是理解 trade-off。

## 6. 不确定性与模型误差

模型给出的一个确定预测可能掩盖真实不确定性。来源包括：

- 训练数据覆盖不足；
- 视觉遮挡；
- 多种未来都可能发生；
- 动力学变化；
- 长期 rollout 漂移。

可用 ensemble、概率模型、预测分歧或保守惩罚估计不确定性。

## 7. 在具身智能中的使用

Model-Based RL 适合：

- 机器人操作；
- 导航与轨迹规划；
- 视觉预测；
- latent dynamics；
- 语言条件规划；
- 数据效率要求高的任务。

真实项目的最小视角：

```text
camera/state/action logs
↓
时间同步与轨迹切分
↓
encoder + dynamics training
↓
one-step / multi-step validation
↓
planner in simulation
↓
real robot short-horizon execution
↓
replan
```

## 8. 常见错误与 Debug

### 8.1 next state 错位

检查 s_t、a_t、s_{t+1} 的时间戳。一个 off-by-one 会让 dynamics 学不到动作影响。

### 8.2 只看 one-step loss

一步预测好不等于多步 rollout 好。必须评估不同 horizon 的误差和行为。

### 8.3 planner 过度相信模型

在分布外区域模型可能自信但错误。加入 uncertainty、真实校正和安全约束。

### 8.4 latent 丢失控制信息

只优化重建可能忽略动作相关状态。增加 action-conditioned prediction 和下游控制评估。

### 8.5 reward/terminal 预测错

奖励时刻、终止条件和 bootstrap 处理必须与环境接口一致。

## 9. 容易混淆

| 概念 | 含义 | 不等于 |
|---|---|---|
| model-based | 使用环境模型规划/学习 | 模型一定准确 |
| model-free | 不显式使用环境转移模型 | 不使用神经网络 |
| dynamics | 状态如何随动作变化 | policy |
| reward model | 预测反馈 | 累计 return |
| imagined rollout | 模型中展开 | 真实交互 |
| MPC | 反复短期规划执行第一步 | 一次生成整条轨迹 |
| latent model | 在表示空间预测 | 必然生成像素 |

## 10. 与其他笔记的关系

前置：[[MDP]]、[[Value_Function]]、[[Actor_Critic]]、[[损失函数]]、[[优化器]]、[[PyTorch]]。

相关：[[视觉表征]]、[[视频理解]]、[[Transformer]]、[[Offline_RL]]、[[Imitation_Learning]]。

本篇不展开具体 World Model 论文的全部架构；先掌握模型、规划、误差和评估的闭环。

## 11. 知识地图

```text
data
↓
encoder
↓
latent dynamics + reward/terminal
↓
imagined rollout
↓
planner / actor / critic
↓
real action
↓
new observation
↓
replan and update
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| transition | s_t、a_t、s_{t+1} 对齐 |
| prediction | latent/observation/reward/terminal |
| horizon | 误差和计算的折中 |
| uncertainty | 分布外风险 |
| evaluation | one-step + multi-step + closed-loop |
| planner | 是否只执行第一步 |
| safety | 约束、回退、急停 |

## 13. 练习题（题目下附参考答案）

1. Model-Based RL 的核心思想是什么？

**参考答案：** 学习或使用环境模型，预测动作后的未来并据此规划或辅助策略训练。

2. dynamics model 预测什么？

**参考答案：** 给定当前状态和动作后的下一状态或 latent。

3. reward model 预测什么？

**参考答案：** 当前动作导致的即时奖励或其他反馈。

4. imagined rollout 是什么？

**参考答案：** 在学习模型内部而非真实环境中展开的未来轨迹。

5. MPC 为什么只执行规划序列第一步？

**参考答案：** 执行后获得新观测再重新规划，减少长期模型误差影响。

6. horizon 太长有什么风险？

**参考答案：** 模型误差逐步累积，规划可能相信不存在的未来。

7. one-step loss 低能保证 multi-step 好吗？

**参考答案：** 不能，递归展开会积累误差，必须单独评估多步预测。

8. World Model 为什么常在 latent space 预测？

**参考答案：** latent 比像素更紧凑，可能更关注任务和动力学相关信息，计算更可控。

9. dynamics 训练的基本样本是什么？

**参考答案：** s_t、a_t、s_{t+1}，视觉任务中还可能包含 o_t、o_{t+1} 和时间信息。

10. next state 错位会怎样？

**参考答案：** 模型把动作与错误未来配对，学不到真实动作影响。

11. 如何估计模型不确定性？

**参考答案：** 使用 ensemble、概率预测、模型分歧或预测误差统计。

12. planner 为什么可能选择危险动作？

**参考答案：** 它在训练分布外区域过度相信模型，模型自信但错误。

13. reward 和 terminal head 为什么重要？

**参考答案：** 规划不仅预测状态，还要知道回报和何时结束。

14. action-conditioned latent 预测有什么价值？

**参考答案：** 让模型学习不同动作导致的未来差异，而不是只预测被动变化。

15. Model-Based RL 如何提高数据效率？

**参考答案：** 通过模型复用真实数据，在 imagined 环境中产生更多候选未来或训练样本。

16. 机器人使用 MPC 要注意什么？

**参考答案：** 规划延迟、动作安全、模型不确定性、真实观测校正和执行第一步后的重规划。

17. 为什么要可视化 rollout？

**参考答案：** 发现模型在空间、运动和任务语义上的预测漂移。

18. World Model 的 latent 重建很好但规划差，为什么？

**参考答案：** latent 可能保留像素细节却丢掉动作和动力学相关信息，目标与规划不匹配。

19. Offline 数据如何支持 Model-Based RL？

**参考答案：** 从固定轨迹学习 dynamics/reward，再在模型中规划，但要防止分布外 rollout。

20. 读 Model-Based RL 代码最先看什么？

**参考答案：** encoder/dynamics 输入输出、时间对齐、rollout、planner、horizon、uncertainty 和真实评估。

## 13. 掌握度检查（附答案）

1. Model-Based 和 Model-Free 的区别是什么？

**答案：** 前者显式使用环境模型预测/规划，后者直接学习策略或价值而不依赖显式转移模型。

2. dynamics model 是什么？

**答案：** 预测状态在动作作用下如何变化的模型。

3. World Model 常包含哪些部分？

**答案：** 观测 encoder、latent dynamics、reward/terminal 预测头，可能还有 decoder 和 planner。

4. imagined rollout 与真实 rollout 的区别是什么？

**答案：** 前者在模型中展开，后者在真实环境中执行并观测。

5. MPC 的核心循环是什么？

**答案：** 预测/规划一段动作，只执行第一步，获得真实新状态后重新规划。

6. 为什么 horizon 存在取舍？

**答案：** 越长看得远但误差和计算增加，越短稳定但可能短视。

7. 为什么 one-step 评估不够？

**答案：** 多步递归会累积模型误差，闭环使用更接近真实需求。

8. 不确定性有什么作用？

**答案：** 提示模型未见区域和风险，帮助保守规划或避免危险动作。

9. latent model 的优点是什么？

**答案：** 紧凑、计算可控并可聚焦任务相关信息。

10. 为什么必须保留时间戳？

**答案：** 对齐状态、动作、观测和 reward，避免 transition 错位。

11. Model-Based RL 在机器人中的价值是什么？

**答案：** 减少真实交互，支持短期预测、规划和安全控制。

12. 规划失败可能是算法错还是模型错？

**答案：** 两者都可能，需要分离评估 dynamics、reward、planner 和真实执行。

13. 何时查 [[视觉表征]]？

**答案：** 需要理解 observation 如何压缩为可预测、可控制 latent 时。

14. 何时查 [[Offline_RL]]？

**答案：** 需要从固定数据学习模型或价值并处理分布外行为时。

15. 读 World Model 代码最先确认哪五项？

**答案：** latent 定义、transition 对齐、reward/terminal、rollout horizon、真实闭环评估。
