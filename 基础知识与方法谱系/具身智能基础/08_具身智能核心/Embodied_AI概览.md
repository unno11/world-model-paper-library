# Embodied AI 概览：让智能体在世界中感知、行动和学习

具身智能（Embodied AI）研究能在物理或仿真环境中感知世界、采取动作并完成任务的智能系统。它不只是“视觉模型加机器人”，而是感知、状态表示、决策、规划、控制和环境反馈的闭环。

本文负责总览和模块地图；具体行为克隆、World Model、VLA、规划与控制分别见本目录其他笔记，机器人几何与执行见 [[Manipulation]]、[[Control]]，强化学习见 [[MDP]]。

## 0. 学习目标

学完以后能够：

- 解释具身智能与普通离线预测任务的差异；
- 画出 observation → action → environment 的闭环；
- 区分 perception、representation、policy、planner、controller；
- 读懂一个具身智能 repository 的整体结构；
- 识别数据、时间、坐标、评估和安全边界；
- 将 CV、RL、机器人和 World Model 连接起来。

## 1. 为什么需要具身智能？

图像分类只需要回答“这是什么”，具身智能还要回答“我现在在哪里、目标在哪里、下一步怎么做、动作后会发生什么”。

```text
环境
↓ observation
智能体内部表示
↓ policy / planner
action
↓
机器人与环境改变
↓
新 observation
└───────────────闭环
```

同一个视觉输入在不同机器人状态、目标和历史下可能需要不同动作。

## 2. 核心模块

### 2.1 Perception

从 RGB、深度、触觉、关节状态等传感器得到可用输入。见 [[图像基础]]、[[视觉表征]]。

### 2.2 Representation

把原始 observation 编码成状态、token 或 latent。好的表示要保留任务、空间、动作和未来相关信息。

### 2.3 Policy

根据 observation/state 直接产生动作或动作分布。见 [[Behavior_Cloning]]、[[Diffusion_Policy]]、[[VLA]]。

### 2.4 Planner

比较候选未来或子目标，产生轨迹/动作计划。见 [[Latent_Planning]]、[[Planning_and_Control]]。

### 2.5 Controller

高频跟踪目标，处理反馈、动力学、限幅和安全。见 [[Control]]。

### 2.6 Environment

环境执行 action，返回新观测、reward、终止和任务状态。见 [[MDP]]。

## 3. 具身智能的整体工作原理

```text
传感器
↓
预处理与同步
↓
视觉/状态 encoder
↓
状态或 latent
├─ policy
├─ planner
└─ value/world model
↓
控制器
↓
机器人执行
↓
反馈、评估、恢复
```

## 4. 一个研究项目如何运行

```text
Git repository
↓
config
↓
dataset / environment
↓
model definition
↓
training loop
↓
checkpoint
↓
offline validation
↓
simulation rollout
↓
real robot evaluation
```

读代码时先找入口脚本、配置、数据集、模型、训练循环和评估脚本。

## 5. 数据与时间

具身数据通常同时包含：

- observation；
- action；
- reward/success；
- joint state；
- camera frame；
- timestamp；
- episode/task metadata。

时间错位、相邻帧泄漏和 action 延迟是常见失败来源。

## 6. 具身智能与深度学习/强化学习

- 视觉模型提供 observation encoder；
- Transformer 建模历史、动作和多模态 token；
- 模仿学习从示范初始化；
- RL 用环境反馈改进；
- World Model 预测动作后的未来；
- planner 选择候选轨迹；
- controller 将高层目标变成安全执行。

## 7. 评估不只是 loss

需要看：

- action prediction；
- success rate；
- collision/safety；
- generalization；
- latency；
- closed-loop rollout；
- 多任务、多场景、多种子；
- 仿真到真实差异。

见 [[模型评估]]。

## 8. 常见错误与 Debug

### 8.1 把具身智能当图像分类

只检查离线 accuracy，忽略动作、时间和闭环执行。

### 8.2 模块接口不清楚

policy 输出是 action、goal pose 还是 latent？controller 期待什么单位和 frame？

### 8.3 训练测试泄漏

同一轨迹、物体或场景同时出现在 train/test。

### 8.4 只看单步预测

闭环连续执行会积累误差，需要 rollout 和任务级评估。

### 8.5 忘记安全边界

学习策略必须经过限幅、碰撞/力监测、回退和急停。

## 9. 与其他笔记的关系

前置：[[05_计算机视觉]] 中的视觉笔记、[[06_强化学习]]、[[07_机器人基础]]。

核心：[[Robot_Learning]]、[[World_Model]]、[[Video_World_Model]]、[[VLA]]、[[Representation_for_Control]]。

专题：[[Behavior_Cloning]]、[[Diffusion_Policy]]、[[JEPA]]、[[Latent_Planning]]、[[Planning_and_Control]]。

## 10. 知识地图

```text
感知
↓
表示
↓
策略/规划
↓
控制
↓
环境
↓
反馈与数据
↓
重新训练/改进
```

## 11. 快速查询表

| 代码位置 | 先问什么 |
|---|---|
| dataset | 返回什么、时间是否对齐 |
| encoder | 输出 feature 还是 latent |
| policy | action 的单位和 frame |
| planner | 预测多长、如何评分 |
| controller | 频率、限幅、安全 |
| evaluator | 是否闭环、如何定义成功 |

## 12. 练习题（题目下附参考答案）

1. 具身智能的核心闭环是什么？

**参考答案：** observation → internal state/representation → policy/planner → action → environment → new observation。

2. perception 做什么？

**参考答案：** 从传感器得到可供决策使用的观测和特征。

3. policy 与 controller 的区别是什么？

**参考答案：** policy 决定做什么，controller 负责高频、稳定和安全地执行目标。

4. planner 做什么？

**参考答案：** 比较未来或候选路径，生成动作计划或子目标。

5. 为什么时间戳重要？

**参考答案：** 对齐观测、动作、奖励和执行时刻，避免因果错位。

6. 为什么离线 loss 不够？

**参考答案：** 不能反映闭环误差、延迟、碰撞、成功率和真实执行。

7. World Model 预测什么？

**参考答案：** 给定当前状态和动作后的未来状态/latent、奖励或终止。

8. VLA 中 V、L、A 通常是什么？

**参考答案：** Vision、Language、Action。

9. 具身项目的入口通常在哪里？

**参考答案：** train/eval/rollout 脚本、配置和环境/数据构造处。

10. 为什么需要控制器？

**参考答案：** 学习策略输出不一定直接可执行，控制器负责反馈跟踪、限幅和安全。

11. 训练测试如何避免泄漏？

**参考答案：** 按轨迹、场景、物体、机器人或任务实体切分。

12. 具身智能中常见 observation 有哪些？

**参考答案：** RGB、深度、触觉、关节状态、末端 pose、目标和历史。

13. 失败恢复为什么重要？

**参考答案：** 真实任务存在遮挡、滑落、碰撞和不可达，需要回退或重新规划。

14. 仿真评估高能证明真实成功吗？

**参考答案：** 不能，还需处理 sim-to-real、传感器和动力学差异。

15. policy action 接到机器人前检查什么？

**参考答案：** shape、单位、frame、范围、频率、延迟和安全保护。

16. World Model 与普通视频预测有什么区别？

**参考答案：** World Model 通常需要动作条件和任务决策价值，不只预测外观。

17. 具身表示要保留什么？

**参考答案：** 空间、状态、对象、动作影响和未来预测相关信息。

18. 为什么要区分 observation 和 state？

**参考答案：** observation 可能不完整，策略需利用历史或状态估计。

19. 如何阅读一个具身 research code？

**参考答案：** 先找入口与 config，再看 dataset/env、model、loss、rollout、评估和安全。

20. 具身智能的最终评估是什么？

**参考答案：** 在明确任务和安全条件下的闭环成功、泛化、效率和稳定性。

## 13. 掌握度检查（附答案）

1. 具身智能与普通预测有什么差异？

**答案：** 具身智能的预测会影响动作，动作又改变环境和未来观测，是闭环决策问题。

2. perception、representation、policy、controller 分别做什么？

**答案：** 感知输入、编码状态、决定动作、执行并稳定跟踪。

3. 为什么环境不可省略？

**答案：** 动作的效果只有在环境反馈中才能验证。

4. 为什么时间对齐是核心？

**答案：** 错位会把错误观测和动作配对，模型仍可能训练但学到错误因果。

5. 为什么要有 planner？

**答案：** 复杂任务需要比较未来、子目标和约束，不一定能单步反应。

6. World Model 的动作条件为何重要？

**答案：** 未来状态由动作影响，非条件预测可能学不到可控动力学。

7. 为什么 controller 不等于 policy？

**答案：** 一个做高层决策，一个做低层执行和反馈稳定。

8. 具身数据切分为什么不能简单随机？

**答案：** 轨迹和场景存在相关性，随机切分会泄漏。

9. 闭环评估关注什么？

**答案：** 成功、碰撞、恢复、延迟、泛化和连续执行效果。

10. VLA 的输入输出是什么？

**答案：** 视觉与语言条件输入，输出动作或动作相关决策。

11. 为什么需要安全层？

**答案：** 学习策略可能越界或异常，真实硬件和人需要保护。

12. 具身智能如何连接机器人学？

**答案：** 机器人学提供 frame、运动学、动力学和控制执行。

13. 何时查 [[Robot_Learning]]？

**答案：** 需要理解数据、训练、部署和学习算法的完整工程流程时。

14. 何时查 [[World_Model]]？

**答案：** 需要理解状态转移、预测未来和模型辅助规划时。

15. 读 embodied research code 最先确认哪五项？

**答案：** observation、action、时间、环境闭环、成功/安全评估。
