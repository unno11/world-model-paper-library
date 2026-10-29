# Robot Learning：让机器人从数据和交互中学习

Robot Learning 是让机器人从示范、环境交互、仿真或历史日志中学习感知、动作和决策的方法集合。它不是单一算法，而是一个从数据采集到真实执行的工程闭环。

本文负责 Robot Learning 的整体流程、数据、训练、评估、部署和方法选择；具体方法链接到 [[Behavior_Cloning]]、[[Diffusion_Policy]]、[[Actor_Critic]]、[[Offline_RL]] 和 [[World_Model]]。

## 0. 学习目标

学完以后能够：

- 区分 imitation、online RL、offline RL、model-based 和 self-supervised；
- 画出 robot learning pipeline；
- 理解 observation/action/trajectory 数据结构；
- 判断训练目标是否与闭环任务一致；
- 读懂 research repository 的训练与 rollout 入口；
- 发现数据泄漏、sim-to-real、时间对齐和安全问题。

## 1. 为什么需要 Robot Learning？

机器人任务包含复杂视觉、接触、动力学和环境变化，手工规则难以覆盖所有情况。学习方法利用数据发现规律，但真实机器人数据昂贵且错误可能危险。

```text
数据/示范/环境
↓
预处理与同步
↓
表示学习
↓
policy / value / world model
↓
仿真或离线验证
↓
真实闭环
↓
失败数据与再训练
```

## 2. 核心概念

### 2.1 Observation

机器人在时刻 t 可看到的数据，如图像、深度、关节状态、末端 pose、触觉和历史。

### 2.2 Action

动作可以是关节位置/速度/力矩、末端 delta pose、夹爪命令、技能或 action chunk。必须明确 frame、单位和执行周期。

### 2.3 Trajectory

trajectory 是按时间排列的 observation、action、reward、next observation、done 和 metadata。

### 2.4 Behavior Policy

生成数据的策略可能是人类、脚本、旧模型或随机探索。数据分布决定学习能否泛化。

### 2.5 Sim-to-Real

在仿真训练、真实部署时，视觉、动力学、接触、延迟和噪声分布不同。域随机化、真实数据微调和安全部署是常见手段。

## 3. 方法选择

| 方法 | 数据/交互 | 优点 | 主要风险 |
|---|---|---|---|
| BC | 专家示范 | 简单、稳定 | 分布偏移 |
| Online RL | 环境交互 | 可超越示范 | 样本贵、安全 |
| Offline RL | 固定轨迹 | 不需在线探索 | OOD action |
| Model-Based | 学模型再规划 | 数据效率 | 模型误差 |
| Self-Supervised | 无标签数据 | 学表示 | 目标不匹配/坍塌 |

## 4. 完整工程流程

```text
collect
↓
inspect
↓
synchronize
↓
split by trajectory/entity
↓
train representation/policy/model
↓
validate offline
↓
simulate rollout
↓
real robot safety test
↓
full task evaluation
```

每一步都需要保存配置、版本、随机种子、数据统计和 checkpoint。

## 5. 从零读训练代码结构

```python
dataset = build_dataset(cfg)
model = build_model(cfg)
optimizer = build_optimizer(model, cfg)

for batch in loader:
    output = model(batch)
    loss = compute_loss(output, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

研究代码中还要找：

- cfg 中的 action horizon、image size、control frequency；
- Dataset 如何切 window；
- checkpoint 保存什么；
- eval 是否闭环；
- policy 输出如何进入环境。

## 6. 数据质量与实验设计

检查：

- episode 是否完整；
- 传感器时间戳是否同步；
- action 是否真实执行；
- 失败轨迹是否保留；
- 训练和测试是否实体隔离；
- 数据归一化是否固定；
- 标签/奖励是否可信。

## 7. 在具身智能中的使用

Robot Learning 的最终目标通常不是最小化一个 loss，而是在目标环境完成任务，同时满足安全、效率和泛化要求。

例如：

```text
视觉 encoder
+ robot state
↓
policy / diffusion policy
↓
action chunk
↓
低层 controller
↓
物体操作
↓
success/recovery
```

## 8. 常见错误与 Debug

### 8.1 离线 loss 低、闭环失败

检查状态分布偏移、动作延迟、action scaling、控制器和成功定义。

### 8.2 训练/测试泄漏

同一物体、场景或轨迹出现在两边，指标虚高。

### 8.3 仿真表现好、真实失败

检查相机、动力学、接触、延迟、噪声和动作接口。

### 8.4 数据窗口错位

打印 t、observation、action、next observation 和时间戳。

### 8.5 安全检查缺失

真实部署前设置动作限幅、工作空间限制、碰撞监控、回退和急停。

## 9. 与其他笔记的关系

前置：[[Embodied_AI概览]]、[[视觉表征]]、[[MDP]]、[[Manipulation]]、[[Control]]。

方法：[[Behavior_Cloning]]、[[Diffusion_Policy]]、[[Offline_RL]]、[[Actor_Critic]]、[[Model_Based_RL]]。

表示与规划：[[Representation_for_Control]]、[[World_Model]]、[[Latent_Planning]]。

## 10. 知识地图

```text
数据
↓
表示
↓
策略/模型
↓
离线评估
↓
仿真
↓
安全真实 rollout
↓
失败分析
↓
再训练
```

## 11. 快速查询表

| 研究问题 | 先查 |
|---|---|
| 数据从哪来 | behavior policy、场景、时间 |
| 输出是什么 | action 类型、单位、频率 |
| loss 是否可靠 | target、mask、分布 |
| 能否部署 | controller、延迟、安全 |
| 是否泛化 | entity/scene/task split |
| 是否成功 | closed-loop success |

## 12. 练习题（题目下附参考答案）

1. Robot Learning 是什么？

**参考答案：** 让机器人从示范、交互、仿真或日志中学习感知、动作和决策。

2. observation 与 action 分别是什么？

**参考答案：** observation 是机器人看到的信息，action 是机器人执行的控制选择。

3. trajectory 包含什么？

**参考答案：** 按时间排列的观测、动作、奖励、后继观测、终止和元数据。

4. BC 的主要数据来源是什么？

**参考答案：** 专家示范或高质量行为轨迹。

5. Offline RL 的主要风险是什么？

**参考答案：** 学习策略选择数据未覆盖的动作，价值估计发生错误外推。

6. Model-Based 方法为什么需要评估 rollout？

**参考答案：** 一步预测准确不代表多步模型稳定。

7. 为什么要按 entity/trajectory 切分？

**参考答案：** 防止同一对象、场景和相邻帧泄漏。

8. action 的三个关键属性是什么？

**参考答案：** 语义/类型、单位和执行频率/时间。

9. 训练代码中先看什么？

**参考答案：** dataset、config、model、loss、optimizer、checkpoint 和 eval。

10. sim-to-real 为什么困难？

**参考答案：** 仿真和真实的视觉、动力学、接触、噪声和延迟不同。

11. 真实部署前为什么需要低层 controller？

**参考答案：** 把高层动作稳定、安全地转换为执行器命令。

12. 离线 loss 低为什么不够？

**参考答案：** 不代表闭环成功、泛化、安全和真实硬件执行。

13. 为什么保留失败轨迹？

**参考答案：** 学习边界、恢复行为和风险，而不只是理想动作。

14. 数据时间戳有什么用？

**参考答案：** 对齐相机、状态、动作、奖励和控制执行。

15. Robot Learning 如何使用视觉表征？

**参考答案：** 将图像编码为视觉 feature/latent，供 policy、model 或 planner 使用。

16. World Model 在 Robot Learning 中做什么？

**参考答案：** 预测动作后的未来状态/latent，支持规划或数据效率提升。

17. 研究代码中 checkpoint 需要保存什么？

**参考答案：** 模型、优化器、调度器、配置、随机状态和必要的数据/版本信息。

18. policy 输出异常时先查什么？

**参考答案：** 输入预处理、输出 range、动作单位/frame、时间和控制接口。

19. 如何分析真实机器人失败？

**参考答案：** 回放 observation、action、frame、延迟、控制、接触、成功定义和环境状态。

20. 读 Robot Learning repository 的顺序是什么？

**参考答案：** 入口/config → dataset/env → model → loss/update → rollout/eval → safety/deployment。

## 13. 掌握度检查（附答案）

1. Robot Learning 的闭环是什么？

**答案：** 数据/环境 → 表示/模型 → 策略/规划 → 控制执行 → 反馈与再训练。

2. 为什么 action 定义必须清楚？

**答案：** 同一个网络数值可能代表位置、速度、力矩或 delta，接口错误会导致危险动作。

3. BC 和 RL 的数据来源有什么差异？

**答案：** BC 主要依赖示范，RL 依赖环境奖励和交互。

4. Offline RL 与 BC 的差异是什么？

**答案：** Offline RL 利用长期 reward/value，可能改进行为但有 OOD 风险。

5. Model-Based 方法的核心风险是什么？

**答案：** 模型在多步或分布外区域产生错误预测，规划过度相信它。

6. 为什么要做仿真后真实部署？

**答案：** 先降低风险并发现策略和接口问题。

7. 为什么要保留元数据？

**答案：** 追踪任务、实体、环境、时间、版本和实验可重复性。

8. 数据泄漏如何产生？

**答案：** 同一轨迹、对象、场景或未来信息同时进入训练和评估。

9. 为什么需要失败分析？

**答案：** 具身任务的错误可能来自感知、表示、规划、控制、接触或环境，不是只看 loss。

10. 闭环成功率与 action accuracy 有何区别？

**答案：** 前者衡量任务完成，后者衡量静态动作拟合。

11. sim-to-real 需要匹配什么？

**答案：** 视觉、动力学、接触、延迟、控制频率和传感器噪声。

12. 低层控制器的职责是什么？

**答案：** 高频跟踪、反馈稳定、限幅和安全执行。

13. 何时查 [[Embodied_AI概览]]？

**答案：** 需要建立模块地图和整体研究流程时。

14. 何时查 [[World_Model]]？

**答案：** 需要学习环境动态并进行 imagined planning 时。

15. 读 Robot Learning research code 最先确认哪五项？

**答案：** 数据分布、action schema、时间同步、训练目标、闭环/安全评估。
