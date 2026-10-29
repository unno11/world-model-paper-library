# Representation for Control：学习真正能驱动动作的表示

控制表示（Representation for Control）关注一个问题：视觉、状态和历史被编码后，是否保留了足够的动作、空间和动力学信息，让策略能够做出正确决策。

通用视觉表征见 [[视觉表征]]，自监督见 [[自监督学习]]，控制执行见 [[Control]]；本文重点负责“表示如何服务控制”和评估标准。

## 0. 学习目标

学完以后能够：

- 区分分类表示与控制表示；
- 解释 action-relevant、spatial、temporal 和 predictive information；
- 读懂视觉 latent 与 robot state 融合代码；
- 理解 control probe、action prediction 和 latent dynamics；
- 发现 pooling、增强、坍塌和数据泄漏造成的控制失效；
- 连接表示、策略、规划和闭环评估。

## 1. 为什么控制需要专门表示？

适合分类的 feature 可能只保留“这是杯子”，但控制还需要：

- 杯子在哪里；
- 抓取方向是什么；
- 距离和遮挡如何；
- 当前动作会怎样改变未来；
- 任务是否接近完成。

```text
observation
↓ encoder
control-relevant representation
├─ policy
├─ value
├─ dynamics
└─ planner
↓
action
```

## 2. 核心概念

### 2.1 Action-Relevant

表示包含能改变动作选择的信息。如果去掉某维后策略无法区分应执行的动作，说明它可能有控制价值。

### 2.2 Spatial Information

位置、方向、距离和目标关系常需保留 spatial feature/token。过早 global pooling 可能损失抓取与导航所需信息。

### 2.3 Temporal Information

速度、接触阶段、动作历史和隐藏状态需要时间信息。单帧 latent 可能不足，需要 frame stack、RNN 或 Transformer。

### 2.4 Predictive Representation

如果 latent 能根据 action 预测未来，说明它可能编码了动力学相关结构。但预测准确仍需验证规划和控制。

### 2.5 Proprioception Fusion

视觉与关节、末端、夹爪状态融合时，需要时间同步、尺度匹配、frame 明确和 batch 对齐。

## 3. 表示学习到控制的流程

```text
RGB/depth + proprioception
↓
encoders
↓
fusion
↓
latent z_t
↓
policy/value/dynamics
↓
action
↓
controller
↓
success/recovery
```

## 4. 从零读融合代码

```python
visual_z = visual_encoder(image)
state_z = state_encoder(robot_state)

z = torch.cat([visual_z, state_z], dim=-1)
action = policy_head(z)
```

检查：

- visual_z/state_z 的 batch 和时间是否一致；
- 是否错误 flatten 了空间或时间；
- 两种模态尺度是否差异巨大；
- encoder 是否冻结；
- action 是否进入低层控制。

## 5. 评估控制表示

可做：

- linear/control probe；
- action prediction；
- next-state prediction；
- value prediction；
- cross-scene transfer；
- closed-loop success；
- perturbation/occlusion test；
- latency and robustness。

最重要的是下游控制，不是单独 embedding 可视化。

## 6. 学习目标与增强

目标可能包括：

- 行为克隆；
- action-conditioned prediction；
- contrastive temporal consistency；
- masked future prediction；
- reward/value prediction；
- JEPA latent prediction。

增强必须保留控制语义。随机翻转可能改变左右，过强 crop 可能移除目标，时间打乱会破坏动力学。

## 7. 在具身智能中的使用

表示可服务于：

- manipulation；
- navigation；
- imitation learning；
- RL；
- World Model；
- latent planning；
- VLA。

同一表示可能同时接 policy 和 dynamics，但两者需求不完全相同，需要多任务评估。

## 8. 常见错误与 Debug

### 8.1 只做分类预训练

表示可能对类别有用但不保留位置、动作和未来信息。

### 8.2 过度 pooling

空间目标和物体关系消失，策略无法定位。

### 8.3 模态不同步

视觉属于 t，state 属于 t-1，融合后动作归因错误。

### 8.4 表示坍塌

z 缺乏样本区分能力，策略输出接近常数。

### 8.5 只看 offline probe

probe 高不代表闭环控制成功，需要真实 rollout 和扰动测试。

## 9. 与其他笔记的关系

前置：[[视觉表征]]、[[自监督学习]]、[[Manipulation]]、[[Control]]、[[MDP]]。

相关：[[JEPA]]、[[World_Model]]、[[Latent_Planning]]、[[VLA]]、[[Planning_and_Control]]。

## 10. 知识地图

```text
raw observation
↓
visual/state encoder
↓
fusion + temporal context
↓
control representation
├─ policy
├─ value
├─ dynamics
└─ planning
↓
controller and task success
```

## 11. 快速查询表

| 表示问题 | 检查 |
|---|---|
| spatial | 是否保留 token/feature map |
| temporal | 是否包含历史和 dt |
| action | 是否 action-conditioned |
| fusion | batch、尺度、时间 |
| transfer | scene/entity split |
| control | rollout/success/robustness |

## 12. 练习题（题目下附参考答案）

1. 控制表示与分类表示有何区别？

**参考答案：** 控制表示还要保留位置、姿态、动力学和动作相关信息，而分类可能只需类别信息。

2. action-relevant 表示是什么？

**参考答案：** 包含能影响动作选择和任务结果的信息。

3. 为什么需要 temporal representation？

**参考答案：** 速度、阶段、接触和隐藏状态不能总由单帧决定。

4. 过度 pooling 有什么风险？

**参考答案：** 丢失空间位置和目标关系。

5. 视觉和 state 融合要检查什么？

**参考答案：** batch、时间、尺度、dtype/device 和 frame。

6. control probe 测试什么？

**参考答案：** 表示是否包含可用于动作或控制预测的信息。

7. 为什么分类 probe 高不等于控制好？

**参考答案：** 分类可分性不代表空间、动力学和动作信息完整。

8. action-conditioned prediction 有什么意义？

**参考答案：** 测试表示是否能表达动作对未来的影响。

9. 为什么增强要考虑控制语义？

**参考答案：** 错误增强会改变左右、位置和动力学标签。

10. 如何检查表示坍塌？

**参考答案：** 看 z 方差、样本距离、action prediction 和下游任务。

11. 模态不同步会怎样？

**参考答案：** policy 学到错误的视觉—状态—动作因果关系。

12. control representation 可以接哪些模块？

**参考答案：** policy、value、dynamics 和 planner。

13. World Model 需要什么表示？

**参考答案：** 能表达状态、动作影响和未来可预测性的 latent。

14. 为什么需要闭环评估？

**参考答案：** 表示的实际价值体现在连续动作和环境反馈中。

15. VLA 需要什么控制表示？

**参考答案：** 能融合视觉、语言、机器人状态并产生空间/动作相关信息。

16. 视觉 encoder 冻结后性能差怎么办？

**参考答案：** 检查预处理和分布，再考虑微调、适配器或任务相关训练。

17. latent 维度大一定更好吗？

**参考答案：** 不一定，过大可能含噪、计算慢和过拟合。

18. 如何评估空间信息？

**参考答案：** dense probe、定位/抓取任务、遮挡与平移扰动测试。

19. 如何评估未来信息？

**参考答案：** next-state/latent prediction、rollout 和规划成功。

20. 读控制表示代码最先看什么？

**参考答案：** 输入模态、输出 shape、时间/空间是否保留、训练目标和闭环用途。

## 13. 掌握度检查（附答案）

1. 控制表示要保留什么？

**答案：** 任务相关的空间、状态、时间、动作和未来信息。

2. 为什么分类 feature 可能不适合控制？

**答案：** 它可能丢失位置、姿态、速度和动作影响。

3. spatial representation 有什么价值？

**答案：** 支持定位、抓取、导航和空间关系判断。

4. temporal representation 有什么价值？

**答案：** 表达速度、历史、接触阶段和部分可观测状态。

5. action-conditioned 表示是什么？

**答案：** 在表示或预测中显式利用动作条件，表达可控未来。

6. 为什么融合要同步？

**答案：** 不同模态错时会导致错误的因果配对。

7. control probe 与 linear probe 差异是什么？

**答案：** control probe 更直接测试动作/控制相关信息，不只测分类可分性。

8. 为什么要做闭环测试？

**答案：** 控制表现受误差累积、延迟和环境反馈影响。

9. 过度 pooling 的代价是什么？

**答案：** 空间精度下降。

10. 为什么增强可能伤害机器人任务？

**答案：** 变换可能改变动作语义和几何关系。

11. representation collapse 对 policy 有何表现？

**答案：** 不同状态产生近似相同动作或策略缺乏区分。

12. World Model 和 policy 对表示的需求相同吗？

**答案：** 不完全相同，前者需要动力学可预测性，后者需要动作区分和执行相关性。

13. 何时查 [[视觉表征]]？

**答案：** 需要理解 global/spatial、invariance 和预训练迁移时。

14. 何时查 [[JEPA]]？

**答案：** 需要通过未来/被遮挡 latent prediction 学表示时。

15. 读控制表示 research code 最先确认哪五项？

**答案：** 模态同步、空间/时间保留、action condition、下游 head、闭环评估。
