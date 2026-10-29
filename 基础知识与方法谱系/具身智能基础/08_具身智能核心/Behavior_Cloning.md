# Behavior Cloning：把专家动作变成可执行策略

行为克隆（Behavior Cloning，BC）是最直接的具身学习方法：把专家在观测下采取的动作当作监督标签，训练 policy 复现它。

前面的 [[Imitation_Learning]] 介绍模仿学习一般问题；本文更聚焦具身智能中的 BC 代码、数据窗口、动作表示和闭环部署。

## 0. 学习目标

学完以后能够：

- 从数据结构理解 BC 训练样本；
- 解释 observation→action 的监督目标；
- 处理视觉、状态和历史窗口；
- 理解 covariate shift 与 compounding error；
- 读懂 BC 训练/评估代码；
- 排查 action 对齐、缩放、切分和闭环失败。

## 1. 为什么需要 BC？

专家示范提供了“在这个状态应该怎么做”的直接信息：

```text
专家轨迹
↓
(o_t, a_t) 样本
↓
policy(o_t) 预测动作
↓
监督 loss
↓
部署 rollout
```

相比从随机探索开始，BC 可以快速得到一个能工作的初始策略，但它只学到数据覆盖范围内的行为。

## 2. 核心概念

### 2.1 Observation

可包含 RGB/depth、关节状态、末端 pose、夹爪状态、语言目标和历史帧。

### 2.2 Action

常见形式：

- joint position/velocity；
- torque；
- end-effector delta pose；
- gripper open/close；
- action chunk。

必须明确单位、frame、时间间隔和是否归一化。

### 2.3 Behavior Cloning Loss

连续动作常用 MSE/Huber，离散动作常用交叉熵：

$$
L(\theta)=E[L(\pi_\theta(o_t),a_t^{expert})]
$$

动作维度不同或部分动作无效时，需要 mask 或加权。

### 2.4 Covariate Shift

训练只看专家轨迹，部署时策略小误差会进入未见状态，错误继续累积。

## 3. 数据窗口与时间对齐

```text
[o_{t-L+1},...,o_t]
↓
encoder / temporal model
↓
预测 a_t 或 [a_t,...,a_{t+H-1}]
```

必须明确：

- 当前 observation 对应当前动作还是下一动作；
- action 是否有执行延迟；
- window 是否跨 episode；
- padding 是否被 mask；
- camera/state/action 是否同步。

## 4. 从零读 BC 代码

```python
for batch in loader:
    obs = batch["obs"]
    target_action = batch["action"]

    pred_action = policy(obs)
    loss = action_loss(pred_action, target_action)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

阅读时继续追踪：

- policy 输入如何构造；
- output 如何 scale 回物理单位；
- DataLoader 是否打乱时间；
- eval 是否真正 rollout；
- checkpoint 是否保存 normalization。

## 5. 视觉 BC

```text
RGB/depth
↓ CNN/ViT encoder
↓
visual feature
+ robot state
↓
MLP/RNN/Transformer policy
↓
action
```

分类准确率不是 BC 的目标，动作预测和闭环任务才是关键。

## 6. 评估与部署

分三层：

1. offline action error；
2. open-loop 短序列预测；
3. closed-loop task success。

部署前还要检查：

- 动作范围和频率；
- 控制器；
- 延迟；
- 碰撞和急停；
- 失败恢复。

## 7. 常见错误与 Debug

### 7.1 action shift 错

用时间戳打印 o_t、a_t、o_{t+1}，确认监督关系。

### 7.2 物理单位错

归一化动作未反归一化、位置/速度混淆或 frame 错。

### 7.3 随机样本切分泄漏

按 trajectory、scene、object 或 robot 切分。

### 7.4 训练 loss 很低、闭环失败

检查状态偏移、观测延迟、控制执行、动作 chunk 和成功定义。

### 7.5 专家质量不一致

过滤失败/异常示范，或增加任务/风格条件。

## 8. 与其他笔记的关系

前置：[[Imitation_Learning]]、[[Robot_Learning]]、[[视觉表征]]、[[Control]]、[[模型评估]]。

相关：[[Diffusion_Policy]]、[[VLA]]、[[Representation_for_Control]]、[[Planning_and_Control]]。

## 9. 知识地图

```text
expert data
↓
sync/window/normalize
↓
policy training
↓
offline action metric
↓
closed-loop rollout
↓
covariate shift
↓
DAgger / recovery / data collection
```

## 10. 快速查询表

| 检查项 | 重点 |
|---|---|
| obs | 当前/历史窗口 |
| action | 时刻、单位、frame |
| split | trajectory/entity |
| loss | 动作类型与 mask |
| scaling | 训练值→物理值 |
| eval | rollout 而非只算 loss |
| safety | 限幅和急停 |

## 11. 练习题（题目下附参考答案）

1. BC 的输入输出是什么？

**参考答案：** 输入专家观测，输出专家在该观测下的动作。

2. BC 为什么是监督学习？

**参考答案：** 专家动作充当标签，直接计算预测动作和目标动作的 loss。

3. covariate shift 是什么？

**参考答案：** 部署时策略进入训练数据未覆盖的状态分布。

4. action chunk 是什么？

**参考答案：** 一次预测连续多个未来动作。

5. 为什么要记录 timestamp？

**参考答案：** 对齐观测、动作和执行延迟。

6. 连续动作常用什么 loss？

**参考答案：** MSE、Huber 或其他回归 loss。

7. 离散动作常用什么 loss？

**参考答案：** 交叉熵等分类 loss。

8. 为什么要按轨迹切分？

**参考答案：** 防止相邻帧和同一任务泄漏到验证集。

9. 视觉 BC 的常见结构是什么？

**参考答案：** 视觉 encoder 加机器人状态，再接 policy head。

10. 动作缩放错误会怎样？

**参考答案：** 离线数值可能正常，但真实执行幅度或单位错误。

11. 如何检查 action shift？

**参考答案：** 回放时间戳，打印 observation/action/next observation 的对应关系。

12. 为什么 offline action error 不等于 success rate？

**参考答案：** 单步拟合不能反映闭环误差累积和任务完成条件。

13. 训练 loss 低、闭环失败怎么查？

**参考答案：** 查分布偏移、延迟、动作接口、控制器、时间对齐和成功标签。

14. BC 如何初始化 RL？

**参考答案：** 先用示范训练策略，再用环境奖励在线改进。

15. 失败示范是否有价值？

**参考答案：** 可以提供边界和恢复信息，但需正确标注或筛选。

16. 视觉表示应保留什么？

**参考答案：** 目标、位置、姿态、距离和动作相关信息。

17. policy 输出 delta pose 后需要什么？

**参考答案：** 通过坐标、IK 和控制器转换为执行命令。

18. 为什么部署要做动作限幅？

**参考答案：** 防止异常预测造成机械或人身风险。

19. BC 中 window 跨 episode 会有什么问题？

**参考答案：** 把两个无关任务的状态拼在一起，破坏时间语义。

20. 读 BC repository 最先看什么？

**参考答案：** 数据字段、时间对齐、动作尺度、模型输出和 rollout 评估。

## 12. 掌握度检查（附答案）

1. BC 的目标是什么？

**答案：** 拟合专家在观测下的动作策略。

2. 为什么 BC 可能闭环失败？

**答案：** 训练只覆盖专家状态，策略偏离后进入未见状态，误差累积。

3. action schema 为什么重要？

**答案：** 它决定模型输出数字如何被解释和执行。

4. 为什么需要历史窗口？

**答案：** 当前观测可能不完整，历史提供速度和任务阶段信息。

5. 为什么轨迹切分比样本切分好？

**答案：** 保证测试任务与训练轨迹独立。

6. visual BC 的数据流是什么？

**答案：** 图像/状态 → encoder → 融合 → policy → action。

7. 为什么要保存 normalization？

**答案：** 部署必须使用与训练一致的输入输出尺度。

8. offline metric 的局限是什么？

**答案：** 它不包含闭环反馈、延迟、控制和任务成功。

9. 为什么需要 DAgger/数据回收？

**答案：** 为策略实际访问的偏离状态补充专家动作。

10. BC 与 Diffusion Policy 的关系是什么？

**答案：** 都可从示范学习动作，Diffusion Policy 用生成分布建模多峰动作和序列。

11. BC 与 VLA 的关系是什么？

**答案：** VLA 可将视觉语言条件接入动作预测，BC 是其常见训练方式之一。

12. 何时查 [[Control]]？

**答案：** 需要理解动作如何被低层跟踪和安全限制时。

13. 何时查 [[Planning_and_Control]]？

**答案：** 需要理解策略、规划与控制如何分工时。

14. 为什么成功标签要延后验证？

**答案：** 抓取闭合不等于物体稳定，需要观察最终任务状态。

15. 读 BC 代码最先确认哪五项？

**答案：** observation、action 时间、单位/frame、loss、闭环评估。
