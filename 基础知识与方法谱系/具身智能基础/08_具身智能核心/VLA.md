# VLA：把视觉、语言与动作连接起来

VLA（Vision-Language-Action）模型把视觉观测、语言指令和机器人动作放入一个统一的学习系统。它试图让模型不仅识别世界，还能根据语言目标在物理环境中执行任务。

本文负责 VLA 的输入输出、token/表示融合、action head、数据与评估；视觉基础见 [[ViT]]、[[视觉表征]]，控制执行见 [[Control]]，行为学习见 [[Behavior_Cloning]]、[[Diffusion_Policy]]。

## 0. 学习目标

学完以后能够：

- 解释 V、L、A 各自承担什么角色；
- 读懂图像 token、语言 token 和机器人状态的融合；
- 区分语言理解、任务规划和低层动作执行；
- 看懂 VLA research code 的数据流和 action schema；
- 理解预训练迁移、动作 token、连续 action 和闭环评估；
- 排查语言指令、时间对齐、空间 grounding 和安全问题。

## 1. 为什么需要 VLA？

机器人任务常需要理解开放世界视觉和人类语言：

```text
视觉：桌上有多个物体
语言：把红色杯子放到盒子里
↓
grounding：找对应物体和目标
↓
planning：确定步骤
↓
action：接近、抓取、搬运、放置
↓
反馈与纠错
```

视觉告诉“看到什么”，语言告诉“做什么”，动作实现“怎么做”。

## 2. 核心概念

### 2.1 Vision

图像、视频、深度和视觉 feature/token，通常由 CNN/ViT encoder 处理。

### 2.2 Language

指令、任务描述、对话上下文或目标条件，通常由 tokenizer 和语言 encoder/decoder 处理。

### 2.3 Action

输出可以是：

- 离散 action token；
- 关节/末端连续动作；
- action chunk；
- skill/subgoal；
- 供 controller 执行的目标。

### 2.4 Grounding

把语言中的对象、位置、关系和动作约束对应到视觉/机器人状态。grounding 错了，语言理解再好也会抓错物体。

### 2.5 Multimodal Fusion

融合方式可能是拼接 token、cross-attention、共享 Transformer 或分层模块。需要追踪 token shape、mask 和模态顺序。

## 3. VLA 的整体流程

```text
image/video
↓ visual encoder
vision tokens
+
language instruction
↓ language tokens
+
robot state/history
↓ multimodal fusion
↓
task/plan/action representation
↓
action head
↓
IK/controller
↓
robot feedback
```

VLA 不一定直接输出电机命令，常输出高层动作或连续 action chunk。

## 4. 从零读 VLA 代码

```python
vision_z = vision_encoder(images)
language_z = language_encoder(input_ids)
state_z = state_encoder(robot_state)

tokens = torch.cat([vision_z, language_z, state_z], dim=1)
hidden = transformer(tokens, attention_mask=mask)
action = action_head(hidden[:, action_positions])
```

要检查：

- vision/language/state 的 batch 和 dtype；
- token 拼接顺序；
- attention mask；
- action positions；
- action head 输出是否与 controller schema 匹配。

## 5. Action Token 与连续动作

离散动作 token 便于使用语言模型架构，但动作精度和连续性可能受限。连续 action head 更直接，但需要处理 normalization、分布和安全范围。

有些系统把连续动作离散化，有些使用 diffusion/action decoder，还有些输出技能或目标 pose。

## 6. 数据与训练

VLA 数据可能包括：

```text
instruction
+ image/video
+ robot state
+ action trajectory
+ success/metadata
↓
synchronized multimodal sample
↓
behavior cloning / sequence loss / auxiliary losses
↓
checkpoint
```

数据必须覆盖语言变化、视觉变化、动作阶段和失败恢复，否则模型容易只记忆模板。

## 7. 在具身智能中的使用

VLA 适合：

- 语言条件操作；
- 多任务机器人；
- 视觉导航；
- 长时序任务；
- 技能选择和子目标生成；
- 与 World Model/planner 联合。

典型层级：

```text
VLA：理解指令、视觉和任务上下文
↓
subgoal/action chunk
↓
planner/IK
↓
controller
↓
robot
```

## 8. 常见错误与 Debug

### 8.1 语言和视觉对象错位

检查指令、图像、任务 ID 和目标实体是否匹配，不能只看语言 loss。

### 8.2 token mask 错

padding、causal 或 cross-modal mask 错误会让模型看不到必要信息或看到未来动作。

### 8.3 action 时间错位

instruction/image at t 对应 action_t 还是未来 chunk，必须由时间戳确认。

### 8.4 grounding 失败

模型选择了相似但错误的物体。需要可视化目标框、注意区域或中间 grounding 结果。

### 8.5 action 输出不可执行

检查单位、frame、范围、频率、IK、controller 和安全层。

## 9. 与其他笔记的关系

前置：[[ViT]]、[[视觉表征]]、[[Transformer]]、[[MDP]]、[[Control]]、[[Manipulation]]。

相关：[[Behavior_Cloning]]、[[Diffusion_Policy]]、[[World_Model]]、[[Planning_and_Control]]、[[Representation_for_Control]]。

本篇不展开具体大型 VLA 模型或预训练数据工程。

## 10. 知识地图

```text
language instruction
+ vision
+ robot state/history
↓
grounding/fusion
↓
task representation
↓
action/skill/subgoal
↓
IK/planner/controller
↓
feedback and recovery
```

## 11. 快速查询表

| 检查项 | 重点 |
|---|---|
| language | tokenizer、目标与任务 ID |
| vision | image shape、token、时间 |
| fusion | token 顺序、mask |
| action | 连续/离散、单位、horizon |
| grounding | 目标对象是否正确 |
| control | IK、频率、限幅、安全 |
| eval | 语言变化、场景变化、闭环 |

## 12. 练习题（题目下附参考答案）

1. VLA 的 V、L、A 分别是什么？

**参考答案：** Vision、Language、Action。

2. Vision 模块提供什么？

**参考答案：** 图像/视频中的视觉 token 或 feature。

3. Language 模块提供什么？

**参考答案：** 任务指令、目标和上下文语义。

4. Action 可以是什么？

**参考答案：** 动作 token、连续动作、action chunk、skill 或子目标。

5. grounding 做什么？

**参考答案：** 将语言中的对象和关系对应到视觉和机器人实体。

6. 为什么 VLA 不一定直接输出电机命令？

**参考答案：** 它常输出高层动作/目标，低层 IK 和 controller 负责安全执行。

7. 多模态 fusion 需要检查什么？

**参考答案：** token 顺序、shape、mask、batch、dtype 和时间。

8. 连续 action 为什么需要 normalization？

**参考答案：** 统一不同动作维度的尺度并稳定训练。

9. 语言和图像错配会怎样？

**参考答案：** 模型可能学到错误对象和任务关系。

10. VLA 如何完成语言指令操作？

**参考答案：** 理解语言目标、grounding 视觉对象、规划步骤并输出动作。

11. 为什么需要 action schema？

**参考答案：** 明确输出是位置、速度、力矩、delta pose 还是技能。

12. 如何检查 action 时间对齐？

**参考答案：** 对照 instruction/image/state/action 的时间戳和执行延迟。

13. token mask 错会怎样？

**参考答案：** 模型可能看不到必要模态、看到 padding 或泄漏未来动作。

14. grounding 失败如何调试？

**参考答案：** 可视化目标区域/对象 ID，检查指令、检测、坐标和数据标注。

15. VLA 如何结合 Diffusion Policy？

**参考答案：** VLA 提供视觉语言条件或高层表示，diffusion decoder 生成动作序列。

16. VLA 如何结合 World Model？

**参考答案：** VLA 提议动作或目标，World Model 预测未来并帮助筛选。

17. 为什么要测试语言改写？

**参考答案：** 判断模型是否理解任务语义而非记忆固定句式。

18. 为什么要测试未见物体/场景？

**参考答案：** 检查视觉和语言泛化，而不是训练场景记忆。

19. action loss 低但抓错对象怎么办？

**参考答案：** 检查 grounding、实体 split、语言/图像对齐和闭环评估。

20. 读 VLA repository 最先看什么？

**参考答案：** 多模态数据结构、token/fusion、action schema、时间、控制接口和评估。

## 13. 掌握度检查（附答案）

1. VLA 的目标是什么？

**答案：** 将视觉和语言条件转换为机器人可执行的动作或任务决策。

2. grounding 为什么关键？

**答案：** 它将抽象语言指令对应到具体视觉对象和空间实体。

3. multimodal fusion 做什么？

**答案：** 让视觉、语言、状态和历史信息在统一计算中交互。

4. action schema 为什么重要？

**答案：** 模型输出只有按正确单位、frame 和时间被解释才可执行。

5. VLA 与 BC 的关系是什么？

**答案：** VLA 可以使用行为克隆从多模态示范学习动作。

6. VLA 与 Diffusion Policy 的关系是什么？

**答案：** VLA 可提供条件/表示，Diffusion Policy 可生成连续动作序列。

7. 为什么 VLA 需要 controller？

**答案：** 高层模型输出不一定稳定或安全，需要低层反馈执行。

8. token mask 有什么作用？

**答案：** 控制模态、padding、因果和未来信息的可见性。

9. VLA 评估为什么不能只看语言 loss？

**答案：** 语言正确不代表对象 grounding、动作、闭环和任务成功。

10. 如何测试语言泛化？

**答案：** 使用改写指令、组合目标和未见语言表达。

11. 如何测试视觉泛化？

**答案：** 使用未见对象、背景、视角、光照和场景。

12. 为什么动作时间对齐困难？

**答案：** 相机、语言、状态和动作具有不同采样频率、延迟和持续时间。

13. 何时查 [[Behavior_Cloning]]？

**答案：** 需要理解多模态示范到动作监督时。

14. 何时查 [[Planning_and_Control]]？

**答案：** 需要把 VLA 输出转为可达、无碰撞和稳定执行时。

15. 读 VLA research code 最先确认哪五项？

**答案：** 模态输入、token/fusion/mask、动作定义、时间对齐、真实闭环评估。
