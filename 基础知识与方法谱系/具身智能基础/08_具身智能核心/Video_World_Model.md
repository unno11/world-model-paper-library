# Video World Model：从视觉序列预测未来世界

Video World Model 用视频帧、动作和状态学习环境的时空变化。它是 World Model 在视觉任务上的具体化，目标不是只做视频生成，而是让未来预测服务于行动、规划或控制。

视频基础见 [[视频理解]]，World Model 总览见 [[World_Model]]，JEPA 和视觉表示见 [[JEPA]]、[[视觉表征]]。

## 0. 学习目标

学完以后能够：

- 区分帧预测、视频生成和 action-conditioned world model；
- 理解视频 token、时间窗口和动作对齐；
- 读懂 frame encoder、temporal dynamics 和 decoder；
- 解释 pixel prediction 与 latent prediction 的取舍；
- 评估多步 rollout、动作条件和闭环价值；
- 排查帧率、时间错位、长序列和分布漂移。

## 1. 为什么需要 Video World Model？

机器人看到的是连续视觉观测，动作会改变未来画面：

```text
过去视频 o_{t-L:t}
+ action a_t
↓
时空 encoder
↓
当前 latent z_t
↓
dynamics
↓
未来 z_{t+1:t+H}
↓
预测/规划/控制
```

单帧分类无法告诉机器人动作后物体是否会移动、遮挡是否消失或抓取是否成功。

## 2. 核心概念

### 2.1 Pixel Prediction

直接预测未来像素，直观但高维、对纹理敏感且多种未来难以用单一图像表示。

### 2.2 Latent Prediction

先把帧编码为 latent，再预测未来 latent，计算更可控，可能更关注任务结构。

### 2.3 Action-Conditioned

模型接收 action，预测该动作导致的未来。没有 action 条件时，模型可能只学环境自然运动。

### 2.4 Temporal Context

过去帧窗口提供速度、遮挡、任务阶段和隐藏状态。窗口长度影响可观测性与计算。

### 2.5 Video Token

每帧可切为 patch token，再加入时间位置，形成 [B,T,N,D] 或展平为 [B,T×N,D]。

## 3. 整体流程

```text
frames + actions + state
↓ time sync / sampling
↓
frame encoder
↓
spatiotemporal representation
↓
action-conditioned dynamics
↓
future frame/latent/reward/done
↓
rollout/planning
```

## 4. 从零读视频 World Model 代码

```python
z = encoder(frames)                 # [B,T,D]
z_pred = dynamics(z[:, -1], actions)
target = encoder(target_frames).detach()

loss = representation_loss(z_pred, target)
loss = loss + reward_loss(reward_head(z_pred), rewards)
```

要确认：

- z 的 T 是否是时间；
- actions 对应哪个时间点；
- target_frames 是否未来；
- encoder 是否共享、冻结或 EMA；
- loss 是否在有效帧上计算。

## 5. Rollout 与误差

用模型自己的预测作为下一步输入，误差会递归累积：

```text
真实 z_t → 预测 z_{t+1}
预测 z_{t+1} → 预测 z_{t+2}
预测 z_{t+2} → ...
```

teacher forcing 训练可能低估部署时的 rollout 难度，需要 scheduled sampling、latent consistency 或短 horizon replan 等设计。

## 6. 机器人视频数据

必须同步：

- RGB/depth；
- action；
- joint state；
- reward/success；
- timestamp；
- camera calibration；
- episode boundary。

帧率和控制频率不同步时，必须明确 action 在多少帧内生效。

## 7. 在具身智能中的使用

Video World Model 可以支持：

- imagined manipulation；
- 视频预测；
- 视觉导航；
- 未来成功/失败预测；
- MPC；
- action-conditioned representation；
- 规划候选轨迹；
- 大量无标签视频预训练。

## 8. 常见错误与 Debug

### 8.1 action 和 frame 错位

打印 frame timestamp、action timestamp、控制延迟和 target index。

### 8.2 训练时看到未来

mask/context 划分错误会产生泄漏，离线 loss 虚低。

### 8.3 只评估视觉相似度

像素或 perceptual similarity 不代表动作可控、空间准确和规划有效。

### 8.4 长视频显存爆炸

减少帧数、空间 token、分辨率或使用分块/层次化时序模型。

### 8.5 预测过于模糊

多种未来平均、loss 不适合或 decoder 容量/数据不足；latent/probabilistic 方法可能更合适。

## 9. 与其他笔记的关系

前置：[[视频理解]]、[[World_Model]]、[[视觉表征]]、[[JEPA]]、[[Transformer]]。

相关：[[Latent_Planning]]、[[Planning_and_Control]]、[[VLA]]、[[Representation_for_Control]]、[[Model_Based_RL]]。

## 10. 知识地图

```text
video frames
↓
sample/sync
↓
spatial encoder
↓
temporal latent
↓
action-conditioned dynamics
↓
future rollout
↓
planner/controller
↓
real observation correction
```

## 11. 快速查询表

| 检查项 | 重点 |
|---|---|
| FPS/dt | 真实时间间隔 |
| clip | context 长度 |
| action | 生效时间和持续长度 |
| latent | spatial/temporal shape |
| target | 未来是否泄漏 |
| rollout | one/multi-step |
| eval | 预测 + 规划 + 闭环 |

## 12. 练习题（题目下附参考答案）

1. Video World Model 预测什么？

**参考答案：** 预测动作条件下未来视觉帧、latent、奖励或终止状态。

2. 为什么需要过去帧？

**参考答案：** 提供运动、速度、遮挡和任务阶段信息。

3. pixel prediction 与 latent prediction 区别是什么？

**参考答案：** 前者直接预测像素，后者先压缩到表示空间再预测。

4. action-conditioned 是什么？

**参考答案：** 将动作作为条件，预测动作影响下的未来。

5. 为什么帧率重要？

**参考答案：** 它决定相邻帧的真实时间间隔和运动语义。

6. [B,T,N,D] 如何理解？

**参考答案：** batch、时间、每帧 token 数、token 表示维度。

7. rollout 为什么会漂移？

**参考答案：** 每一步预测误差作为下一步输入，误差递归积累。

8. teacher forcing 有什么风险？

**参考答案：** 训练使用真实历史，部署使用预测历史，产生训练/推理差异。

9. 视频数据要同步哪些字段？

**参考答案：** 帧、动作、状态、reward、时间戳和 episode 边界。

10. 为什么不能只看像素相似度？

**参考答案：** 视觉相似不代表动作后果、空间关系和规划价值正确。

11. 未来泄漏如何检查？

**参考答案：** 追踪 context、target 和时间 mask，确认未来帧未进入输入。

12. 模糊预测可能为什么？

**参考答案：** 多个可能未来被平均、目标/loss 不匹配或数据不足。

13. 如何降低长视频计算？

**参考答案：** 减少帧/token、分辨率，使用窗口或层次化模型。

14. Video World Model 如何支持 MPC？

**参考答案：** 在模型中预测不同动作序列的未来，再选择回报/风险更好的动作。

15. 机器人 action 可能跨多少帧？

**参考答案：** 取决于控制频率和动作保持时间，必须从时间戳确认。

16. 未来 latent 应如何评估？

**参考答案：** one/multi-step 预测、动作条件、规划和闭环任务。

17. JEPA 与 Video World Model 如何连接？

**参考答案：** JEPA 可用未来时空表示预测作为训练目标。

18. 为什么接触场景难？

**参考答案：** 视觉遮挡、碰撞、摩擦和多种未来使预测不确定。

19. 训练 loss 低但实际 rollout 差怎么查？

**参考答案：** 查 teacher forcing、action shift、帧率、分布偏移和多步指标。

20. 读视频 World Model 代码最先看什么？

**参考答案：** 时间采样、action 对齐、encoder/dynamics、target、rollout 和规划评估。

## 13. 掌握度检查（附答案）

1. Video World Model 与普通视频模型差异是什么？

**答案：** 它强调动作条件、未来状态和决策/控制价值，而不只是分类或生成。

2. 为什么 action 对齐困难？

**答案：** 相机帧、动作执行和控制频率可能不同，存在延迟和持续时间。

3. latent prediction 为什么有吸引力？

**答案：** 比像素低维，更容易聚焦任务和动态结构。

4. rollout drift 是什么？

**答案：** 递归预测中误差随时间累积，预测逐渐偏离真实未来。

5. 如何评估多步模型？

**答案：** 比较不同 horizon 的预测、奖励、规划和闭环成功。

6. teacher forcing 与部署差异是什么？

**答案：** 训练使用真实前一帧/latent，部署使用模型自己的预测。

7. 视频 token 为什么昂贵？

**答案：** 时间乘空间 patch 后 token 数很大，attention 和激活成本上升。

8. 为什么未来泄漏会产生虚假好结果？

**答案：** 模型直接看到答案，loss 低但没有学会预测。

9. 视频 World Model 如何服务机器人？

**答案：** 预测动作后视觉/状态变化，支持规划、控制和失败预判。

10. 为什么需要真实观测重新校正？

**答案：** 真实反馈可以纠正模型 rollout 漂移。

11. 预测模糊不一定是代码 bug 吗？

**答案：** 不一定，可能是多模态未来、loss 平均或目标表示不适合。

12. 何时查 [[视频理解]]？

**答案：** 需要理解采样、clip、FPS、时间模型和视频 shape 时。

13. 何时查 [[JEPA]]？

**答案：** 需要在时空表示空间预测未来而非像素时。

14. 何时查 [[Planning_and_Control]]？

**答案：** 需要将预测未来转成动作和真实执行时。

15. 读 Video World Model 代码最先确认哪五项？

**答案：** frame/action 时间、token shape、target mask、rollout、闭环评估。
