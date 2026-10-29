# Latent Planning：在表示空间中规划未来

Latent Planning 是在低维、可预测的 latent state 空间中搜索未来动作或子目标，而不是直接在像素空间规划。它将视觉表示、World Model、价值函数和控制连接起来。

World Model 见 [[World_Model]]，表示见 [[Representation_for_Control]]，控制与执行见 [[Planning_and_Control]]、[[Control]]。

## 0. 学习目标

学完以后能够：

- 解释为什么在 latent 空间规划；
- 区分 latent state、observation、goal 和 action；
- 读懂 imagined rollout 和 latent cost；
- 理解 horizon、beam/CEM、value 和 uncertainty；
- 判断 latent 是否适合规划；
- 调试状态坍塌、模型漂移、目标不可达和动作执行错误。

## 1. 为什么需要 Latent Planning？

像素空间维度高、预测成本大且细节噪声多。若 encoder 把观测压缩成 task-relevant latent：

```text
observation
↓ encoder
z_t
↓ dynamics(z_t,a_t)
predicted z_{t+1:t+H}
↓ cost/value
选择动作序列
↓ controller
```

规划在 latent 中更高效，但前提是 z 保留规划所需的信息。

## 2. 核心概念

### 2.1 Latent State

latent state 是用于描述和预测当前情境的内部向量/ token。它可能不等于完整物理状态。

### 2.2 Goal Representation

目标可以是目标图像、目标 latent、语言条件、目标 pose 或成功分类器。规划必须定义“接近目标”如何在 latent 中衡量。

### 2.3 Dynamics

dynamics 预测：

$$
z_{t+1}=f(z_t,a_t)
$$

### 2.4 Cost/Value

规划可最小化 latent distance、动作成本、碰撞风险，也可最大化 predicted return/value。

### 2.5 Horizon

horizon 是向未来展开的步数。太短会短视，太长会累积模型误差。

## 3. 规划方法直觉

- random shooting：随机采样动作序列，选最好；
- beam search：保留若干高分候选；
- CEM：从动作分布采样、筛选精英、更新分布；
- actor proposal + value：策略提议、价值评价；
- gradient planning：若 dynamics 可微，直接优化动作。

选择方法取决于动作空间、模型可微性、计算预算和安全约束。

## 4. 从零读 latent rollout 代码

```python
z = encoder(obs)
candidate_z = z

for t in range(horizon):
    candidate_z = dynamics(candidate_z, candidate_action[t])
    score = value(candidate_z)

best_action = select_best(candidate_actions, score)
```

实际代码常有 batch 个候选动作序列，需要检查 candidate、horizon、latent 和 score 的 shape。

## 5. Latent cost 的风险

latent 距离小不一定代表真实任务接近：

- 表示忽略空间；
- 表示对任务变化不敏感；
- 视觉相似但接触状态不同；
- latent dynamics 不准确；
- 目标表示与当前观测分布不同。

必须用真实任务、几何和闭环指标校验。

## 6. 在具身智能中的使用

Latent planning 可用于：

- manipulation goal reaching；
- 视觉导航；
- 多步动作选择；
- World Model planning；
- VLA 的高层子目标；
- offline 数据中的 imagined planning；
- hierarchical robot learning。

## 7. 常见错误与 Debug

### 7.1 latent 丢失必要信息

规划无法区分不同物体位置或接触状态。做 dense/control probe 和任务 rollout。

### 7.2 dynamics 漂移

短期预测好、长 horizon 崩溃。缩短 horizon、replan、加 uncertainty 或改进模型。

### 7.3 goal latent 不可比

目标和当前 observation 使用不同预处理/encoder，距离没有意义。

### 7.4 planner 选到不可执行动作

规划只看 latent score，没有 IK、collision、速度或安全约束。

### 7.5 candidate shape 错

时间、候选数、动作维和 batch 混淆，模型可能错误广播。

## 8. 与其他笔记的关系

前置：[[World_Model]]、[[Representation_for_Control]]、[[MDP]]、[[Value_Function]]、[[Control]]。

相关：[[JEPA]]、[[Video_World_Model]]、[[Planning_and_Control]]、[[VLA]]、[[Offline_RL]]。

## 9. 知识地图

```text
current observation
↓ encoder
current latent
↓ candidate action sequences
latent dynamics rollout
↓
cost/value/uncertainty
↓
select/replan
↓
controller
↓
new observation
```

## 10. 快速查询表

| 检查项 | 重点 |
|---|---|
| z | 是否保留规划信息 |
| goal | 如何表示、是否同 encoder |
| action candidates | shape、范围、时间 |
| horizon | 短视/漂移 |
| score | cost/value/safety |
| constraints | IK、collision、limits |
| replan | 是否用真实反馈 |

## 11. 练习题（题目下附参考答案）

1. Latent Planning 是什么？

**参考答案：** 在 latent 表示空间中预测未来并选择动作或子目标。

2. 为什么不直接在像素规划？

**参考答案：** 像素维度高、细节噪声多、预测和搜索成本大。

3. latent dynamics 做什么？

**参考答案：** 预测当前 latent 在动作作用下的下一 latent。

4. horizon 是什么？

**参考答案：** 向未来展开的步数。

5. 为什么 horizon 太长有风险？

**参考答案：** 模型误差递归累积，规划会相信错误未来。

6. goal latent 如何使用？

**参考答案：** 作为目标表示，与候选未来 latent 计算距离或价值。

7. random shooting 做什么？

**参考答案：** 随机采样动作序列并选评分最好的。

8. CEM 的直觉是什么？

**参考答案：** 反复采样、筛选精英并更新动作分布。

9. planner 为什么需要 uncertainty？

**参考答案：** 识别模型不确定的未来，避免选择高风险 OOD 方案。

10. latent distance 小就一定成功吗？

**参考答案：** 不一定，latent 可能忽略空间、接触或任务条件。

11. goal 和当前 latent 不可比可能为什么？

**参考答案：** encoder、normalize、视角或语义定义不一致。

12. 规划动作为什么还要经过 controller？

**参考答案：** latent 规划输出不是直接硬件命令，需要可执行、稳定和安全的转换。

13. candidate shape 要包含哪些维度？

**参考答案：** 候选数、时间 horizon、动作维，及可能的 batch 维。

14. latent planning 如何连接 World Model？

**参考答案：** World Model 提供 latent dynamics 和 reward/value 预测。

15. 如何验证 latent 是否适合规划？

**参考答案：** 做多步预测、goal reach、control probe、跨场景和闭环成功评估。

16. planner 选到 IK 无解动作怎么办？

**参考答案：** 将可达性/IK 约束加入规划评分或过滤候选。

17. 为什么实时性重要？

**参考答案：** 规划延迟过大，状态已变化，动作基于过时信息。

18. Offline RL 如何使用 latent planning？

**参考答案：** 从固定数据学习 latent model，在模型内搜索，但要防 OOD rollout。

19. 规划失败如何定位？

**参考答案：** 分别检查表示、dynamics、goal/cost、候选动作、约束和 controller。

20. 读 latent planning 代码最先看什么？

**参考答案：** latent 定义、goal、dynamics、candidate/horizon、score、约束和 replan。

## 12. 掌握度检查（附答案）

1. Latent Planning 的核心优势是什么？

**答案：** 在更低维、更任务相关的空间进行未来预测和搜索。

2. latent state 与 observation 的区别是什么？

**答案：** observation 是原始观测，latent 是模型编码后的内部表示。

3. 为什么 latent 必须保留规划信息？

**答案：** 规划依赖空间、动作后果、目标和成功状态，丢失后距离无意义。

4. horizon 的取舍是什么？

**答案：** 长 horizon 看得远但误差大，短 horizon 稳定但可能短视。

5. cost、value、uncertainty 各做什么？

**答案：** cost 衡量代价，value 估计收益，uncertainty 衡量预测可靠性。

6. CEM 的基本循环是什么？

**答案：** 采样、评估、筛选精英、更新分布、重复。

7. 为什么需要 replan？

**答案：** 用新真实观测修正模型和规划误差。

8. latent distance 为什么可能误导？

**答案：** 表示可能忽略空间、接触、动作和任务状态。

9. 为什么要加入 IK/collision 约束？

**答案：** 规划未来必须可执行且安全。

10. candidate action shape 为什么重要？

**答案：** 错误广播会把候选、时间和动作维度混淆。

11. Latent Planning 如何支持操作？

**答案：** 搜索能让物体/末端接近目标 latent 的动作序列，再经控制执行。

12. Latent Planning 如何支持导航？

**答案：** 预测未来环境/位姿表示并比较路径目标和风险。

13. 何时查 [[World_Model]]？

**答案：** 需要理解 latent dynamics 和 reward/terminal prediction 时。

14. 何时查 [[Planning_and_Control]]？

**答案：** 需要把 latent 计划转成真正可执行动作时。

15. 读 latent planning research code 最先确认哪五项？

**答案：** latent/goal、dynamics、horizon、score/uncertainty、约束和真实反馈。
