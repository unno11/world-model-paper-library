# JEPA：在表示空间预测世界的结构

JEPA（Joint Embedding Predictive Architecture）是一类在表示空间进行预测的架构。它通常让 context encoder 根据可见上下文得到表示，再由 predictor 预测目标区域或未来的 target representation，而不是逐像素重建目标。

本文负责 JEPA 的直觉、context/target/predictor、mask、stop-gradient、时空预测和具身智能联系；自监督总览见 [[自监督学习]]，视觉表示见 [[视觉表征]]，World Model 见 [[World_Model]]。

## 0. 学习目标

学完以后能够：

- 解释 JEPA 为什么在 latent/embedding space 预测；
- 区分 context encoder、target encoder 和 predictor；
- 读懂 masked spatiotemporal prediction 数据流；
- 理解 target 分支更新和 stop-gradient；
- 判断目标泄漏、表示坍塌和下游不匹配；
- 将 JEPA 与视频、机器人和 World Model 连接起来。

## 1. 为什么需要 JEPA？

像素重建要求模型预测每个细节，包括可能与任务无关的纹理和噪声。JEPA 的基本想法是：

```text
context observation
↓ context encoder
context representation
↓ predictor
predicted target representation
↑
target observation → target encoder
↓
representation-space loss
```

它希望模型学习可预测、抽象且与世界结构相关的信息。

## 2. 核心概念

### 2.1 Context

context 是模型可以看到的部分，例如图像中的可见 patch、视频过去帧或机器人历史状态。

### 2.2 Target

target 是需要预测的目标区域、未来帧或未来 latent。它必须从 context 中正确隐藏或时间上不可见。

### 2.3 Context Encoder

编码可见上下文，得到 context embedding。它通常是主要学习分支。

### 2.4 Target Encoder

编码目标内容，产生 target embedding，作为预测目标。它可能是 stop-gradient 分支、指数移动平均更新分支或其他稳定目标机制。

### 2.5 Predictor

接收 context embedding 和目标位置/时间信息，预测 target embedding。

### 2.6 Representation-Space Loss

比较 predicted target representation 与 target representation，如 MSE、cosine 或其他表示距离。具体损失见 [[损失函数]]。

## 3. JEPA 的整体流程

```text
原始图像/视频/轨迹
↓
构造 context 与 target
↓
context encoder
↓
predictor(context, target position)
↓
target encoder(target)
↓
表示空间 loss
↓
更新 context/predictor
↓
稳定更新 target encoder
```

mask、时间和 target 更新策略是读代码时最重要的三件事。

## 4. 从零读代码

```python
context_z = context_encoder(context)
with torch.no_grad():
    target_z = target_encoder(target)

pred_z = predictor(context_z, target_positions)
loss = mse_loss(pred_z, target_z)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

要确认：

- target 是否真的不在 context；
- target_z 是否 stop-gradient；
- target encoder 是否由 EMA 更新；
- pred/target shape 和位置是否匹配；
- loss 是否只对目标区域计算。

## 5. Mask 与时空结构

视觉 JEPA 可 mask patch；视频/机器人 JEPA 可 mask 未来时间块或空间区域。mask 定义了学习任务：

- 空间 mask：理解被遮挡区域；
- 时间 mask：预测未来；
- 轨迹 mask：根据历史预测目标状态；
- 多视角 mask：从一个视角预测另一个视角。

mask 过于容易会导致复制，过于困难会让目标不可预测。

## 6. JEPA 与 World Model

如果 target 是未来 latent，JEPA 可成为动作条件世界模型的一部分：

```text
当前视觉/状态 context
+ action
↓ predictor
未来 target latent
↓
规划/控制
```

关键区别是：普通表示预训练只需学习有用特征，World Model 还需要表达动作如何影响未来。

## 7. 防止坍塌与评估

表示坍塌表现为不同输入得到相同 z。检查：

- batch 内 z 方差；
- 样本间余弦相似度；
- target encoder 更新；
- predictor 是否走捷径；
- linear probe、迁移和多步预测。

不能只看 representation loss。

## 8. 在具身智能中的使用

JEPA 类目标可用于：

- 视频未来表示预测；
- 机器人观测历史预测；
- 视觉和状态联合表示；
- action-conditioned latent dynamics；
- 规划前的状态表征预训练；
- 从大量无标签机器人日志学习。

## 9. 常见错误与 Debug

### 9.1 target 泄漏

context 中包含目标 patch 或未来帧，loss 虚低但任务无意义。

### 9.2 target encoder 错误更新

误将 target encoder 放入 optimizer，或 EMA 系数/更新时机错误。

### 9.3 predictor 只记位置

目标位置和数据规律过于简单，模型没学习内容。

### 9.4 目标不匹配下游

预测视觉细节但控制需要动作/空间信息，线性探测和闭环性能差。

### 9.5 mask 维度错

空间 patch、时间步、batch 和 head 维错误，可能造成隐性数据泄漏。

## 10. 与其他笔记的关系

前置：[[自监督学习]]、[[视觉表征]]、[[ViT]]、[[Transformer]]、[[PyTorch]]。

相关：[[World_Model]]、[[Video_World_Model]]、[[Representation_for_Control]]、[[Latent_Planning]]。

本篇不展开某一具体 JEPA 论文的全部实现细节。

## 11. 知识地图

```text
context/target construction
↓
context encoder
↓
predictor
↓
target encoder
↓
latent loss
↓
防泄漏/防坍塌
↓
表示、动力学、规划
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| context | 能看到哪些信息 |
| target | 预测什么、是否隐藏 |
| target encoder | detach/EMA/optimizer |
| predictor | 是否使用位置/时间/action |
| loss | shape、mask、尺度 |
| evaluation | probe、预测、闭环 |

## 13. 练习题（题目下附参考答案）

1. JEPA 的核心预测空间是什么？

**参考答案：** 表示/embedding/latent 空间，而不是必须逐像素预测。

2. context 是什么？

**参考答案：** 模型可见的上下文输入。

3. target 是什么？

**参考答案：** 被隐藏或未来的目标内容及其表示。

4. predictor 做什么？

**参考答案：** 根据 context 表示预测 target 表示。

5. target encoder 为什么可能 stop-gradient？

**参考答案：** 让目标表示稳定，避免两分支互相追逐和坍塌。

6. mask 为什么重要？

**参考答案：** 定义模型必须从上下文推断什么，防止直接复制目标。

7. JEPA 与像素重建的差异是什么？

**参考答案：** JEPA 比较抽象表示，像素重建直接比较原始图像细节。

8. target encoder 可以怎样更新？

**参考答案：** 可能固定、EMA 更新或按具体算法设计更新。

9. 如何检查 target 泄漏？

**参考答案：** 追踪 context 数据，确认目标区域/未来帧未被输入。

10. 表示坍塌如何发现？

**参考答案：** 检查表示方差、相似度、线性探测和下游任务。

11. action-conditioned JEPA 多了什么？

**参考答案：** 将动作作为条件，预测动作影响下的未来表示。

12. 为什么目标位置/时间要输入 predictor？

**参考答案：** 同一 context 下不同空间或未来位置的目标不同。

13. JEPA 如何用于视频？

**参考答案：** 用过去/可见帧表示预测未来帧或被 mask 的时空表示。

14. JEPA 如何用于机器人日志？

**参考答案：** 从历史观测和动作预测未来状态/latent。

15. loss 低但 probe 差可能为什么？

**参考答案：** 发生坍塌、目标太容易、mask 泄漏或表示与任务不匹配。

16. target encoder 放入 optimizer 可能有何风险？

**参考答案：** 破坏稳定目标机制，使两分支同时追逐。

17. 为什么 mask 太容易不好？

**参考答案：** 模型可能通过局部复制完成任务，没有学习结构。

18. 为什么 mask 太难也不好？

**参考答案：** 目标不可预测，梯度信号弱或训练不稳定。

19. JEPA 的 latent 能直接用于控制吗？

**参考答案：** 需要下游验证，表示必须保留空间、动作和动力学信息。

20. 读 JEPA 代码最先看什么？

**参考答案：** context/target 构造、mask、两套 encoder、predictor、loss 和更新路径。

## 13. 掌握度检查（附答案）

1. JEPA 的基本思想是什么？

**答案：** 从可见上下文预测不可见/未来目标的表示。

2. 为什么在表示空间预测？

**答案：** 减少对像素细节的依赖，更关注抽象结构和可预测信息。

3. context encoder 与 target encoder 区别是什么？

**答案：** 前者编码上下文并通常通过梯度学习，后者产生稳定目标表示。

4. predictor 的输入输出是什么？

**答案：** 输入 context 表示及位置/时间/动作条件，输出 target 表示预测。

5. stop-gradient 做什么？

**答案：** 阻断梯度通过目标分支传播。

6. EMA target encoder 的直觉是什么？

**答案：** 用较平滑的历史参数构造稳定目标。

7. mask 的核心作用是什么？

**答案：** 让目标不可直接访问，定义预测任务。

8. 为什么不能只看 JEPA loss？

**答案：** 低 loss 可能来自泄漏、坍塌或简单复制，需看表示和下游。

9. JEPA 与 World Model 的联系是什么？

**答案：** 未来 latent prediction 可作为 action-conditioned dynamics 的表示目标。

10. JEPA 与控制表示有什么区别？

**答案：** JEPA 是训练方法，控制表示关注是否保留动作相关空间/动力学信息。

11. 如何检查坍塌？

**答案：** 看 z 方差、相似度、probe 和下游预测。

12. 为什么 action condition 重要？

**答案：** 未来状态取决于动作，预测必须建模可控变化。

13. 何时查 [[自监督学习]]？

**答案：** 需要比较 masked、对比、时序等自监督范式时。

14. 何时查 [[World_Model]]？

**答案：** 需要把 latent prediction 放入状态转移、规划和控制闭环时。

15. 读 JEPA research code 最先确认哪五项？

**答案：** mask、target 泄漏、encoder 更新、predictor 条件、下游评估。
