# World Model：学习环境如何随动作变化

World Model 是对环境状态、观测、奖励和终止变化的可学习模型。它的目标不是只生成好看的图，而是帮助智能体预测、规划、想象和控制。

本文负责通用 World Model 的组成、latent dynamics、action conditioning、rollout、不确定性和评估；Model-Based RL 基础见 [[Model_Based_RL]]，视频和 JEPA 专题见 [[Video_World_Model]]、[[JEPA]]。

## 0. 学习目标

学完以后能够：

- 解释 World Model 与普通预测模型的差异；
- 区分 observation、state、latent、dynamics、reward model；
- 读懂 encoder → dynamics → decoder/head 的代码；
- 理解 action-conditioned rollout 和误差积累；
- 设计 one-step、multi-step、planning 和闭环评估；
- 排查时间错位、latent 坍塌、分布外 rollout 和奖励错误。

## 1. 为什么需要 World Model？

真实环境交互贵且危险。若模型能预测动作后的未来，智能体可以先在内部比较候选方案：

```text
当前观测 o_t
↓ encoder
latent z_t
↓ dynamics(z_t, a_t)
预测 z_{t+1}, r_{t+1}, done
↓ 重复 rollout
候选未来
↓ planner/policy
选择动作
```

模型预测不是现实本身，错误会随着 rollout 累积。

## 2. 核心概念

### 2.1 Observation 与 State

observation 是感知输入，state 是理想完整状态。World Model 可能直接预测 observation，也可能先编码为 latent。

### 2.2 Latent State

latent 是压缩后的内部状态。好的 latent 要保留未来、动作影响和任务相关信息，而不是只保留像素外观。

### 2.3 Dynamics Model

学习：

$$
z_{t+1}=f_\theta(z_t,a_t)
$$

也可能输出分布、不确定性或多个可能未来。

### 2.4 Reward/Termination Model

预测动作后的 reward 和 done，使 planner 能判断未来收益和任务是否结束。

### 2.5 Decoder

decoder 将 latent 重建为 observation，可用于辅助训练和可视化，但重建好不等于控制好。

### 2.6 Planning

在模型中展开候选动作，比较 predicted return、成功概率、风险或约束，再执行真实第一步。

## 3. World Model 的整体工作原理

```text
o_t
↓ encoder
z_t
+ action a_t
↓ dynamics
z_{t+1}
├─ reward head
├─ terminal head
└─ decoder/observation head
↓
planner / actor / critic
↓
real action
↓
new observation
```

## 4. 从零读代码

```python
z = encoder(obs)
next_z = dynamics(z, action)
pred_reward = reward_head(next_z)
pred_done = done_head(next_z)

loss = state_loss(next_z, target_z)
loss = loss + reward_loss(pred_reward, reward)
loss = loss + done_loss(pred_done, done)
```

重点追踪：

- target_z 如何得到；
- action 作用在当前还是下一状态；
- reward 的时间定义；
- done 是否用于 bootstrap；
- 多步 rollout 是否 detach 或反向传播。

## 5. One-step 与 Multi-step

one-step loss 检查局部转移；multi-step rollout 检查模型递归使用时是否漂移。两者都必须报告。

多步误差来源：

- 每步预测误差；
- latent 分布逐渐偏离训练分布；
- 不确定性被忽略；
- 动作条件错误；
- 视觉观测遮挡。

## 6. World Model 与规划/控制

World Model 可支持：

- MPC；
- imagined policy training；
- value prediction；
- goal-conditioned planning；
- latent search；
- data-efficient robot learning。

规划器应考虑模型不确定性、动作限制、碰撞和真实观测校正。

## 7. 在具身智能中的使用

```text
RGB/depth/state/action logs
↓
时间同步与切分
↓
encoder/dynamics training
↓
multi-step latent rollout
↓
simulation/planning
↓
短 horizon real execution
↓
re-observe/replan
```

机器人世界中物体接触和遮挡会使确定性模型很快失效，因此评估需包含接触、失败和分布外场景。

## 8. 常见错误与 Debug

### 8.1 action 与 next state 错位

打印 t、a_t、o_{t+1} 时间戳，确认环境接口。

### 8.2 latent 坍塌

不同观测得到相同 z，可能是损失、target、decoder 或训练路径有问题。

### 8.3 只看重建 loss

模型可能关注纹理而忽略动作可预测性。加入 action-conditioned 和下游控制评估。

### 8.4 rollout 越长越不可信

缩短 horizon、加入 uncertainty、replan 或使用更好的状态表示。

### 8.5 planner 选择 OOD 未来

检查候选动作是否在数据覆盖内，使用保守评分和安全约束。

## 9. 与其他笔记的关系

前置：[[MDP]]、[[Model_Based_RL]]、[[视觉表征]]、[[Transformer]]、[[JEPA]]。

相关：[[Video_World_Model]]、[[Latent_Planning]]、[[Representation_for_Control]]、[[Planning_and_Control]]、[[VLA]]。

## 10. 知识地图

```text
observation
↓ encoder
latent
↓ action-conditioned dynamics
future latent/reward/done
↓ rollout
planner/value/policy
↓
real control
↓
feedback and model update
```

## 11. 快速查询表

| 检查项 | 重点 |
|---|---|
| latent | 包含什么任务信息 |
| dynamics | action 条件与时间 |
| target | next 时刻和 detach |
| rollout | one/multi-step |
| uncertainty | 预测分歧/风险 |
| planner | horizon、约束、replan |
| eval | latent、任务、闭环 |

## 12. 练习题（题目下附参考答案）

1. World Model 学习什么？

**参考答案：** 学习环境状态/latent、奖励和终止如何随动作变化。

2. latent state 是什么？

**参考答案：** 对观测进行压缩后用于预测和决策的内部表示。

3. dynamics model 输入输出是什么？

**参考答案：** 当前 latent 和 action，输出下一 latent 或其分布。

4. 为什么 action conditioning 必须存在？

**参考答案：** 动作会改变未来，模型需要区分不同动作造成的后果。

5. reward head 做什么？

**参考答案：** 预测未来即时奖励或任务反馈。

6. one-step 与 multi-step 有何区别？

**参考答案：** one-step 检查单次转移，multi-step 检查递归展开和误差积累。

7. decoder 重建好能说明控制好吗？

**参考答案：** 不能，像素重建与动作相关信息不一定一致。

8. planner 为什么要考虑 uncertainty？

**参考答案：** 模型在未见区域可能错误，uncertainty 帮助规避风险。

9. 为什么使用 MPC？

**参考答案：** 执行第一步后重新观测和规划，减少长期模型误差。

10. next state 错位如何检查？

**参考答案：** 对齐 observation、action、next observation 的时间戳和索引。

11. latent 坍塌怎么发现？

**参考答案：** 检查不同样本 latent 方差、距离和下游任务能力。

12. World Model 如何帮助 Offline RL？

**参考答案：** 从固定数据学 dynamics，在模型中辅助规划或产生短期想象数据。

13. World Model 如何帮助 Robot Learning？

**参考答案：** 减少真实试错，预测动作后果并支持策略/规划。

14. 为什么接触场景难预测？

**参考答案：** 接触、摩擦、遮挡和多种未来使动力学不连续或不确定。

15. rollout 很快漂移时查什么？

**参考答案：** 表示、action 对齐、训练分布、horizon、uncertainty 和模型容量。

16. 为什么需要 terminal head？

**参考答案：** 规划需要知道成功、失败或 episode 何时结束。

17. 视觉 World Model 输入可能包括什么？

**参考答案：** RGB/depth、状态、动作、历史和目标条件。

18. latent planning 与像素规划相比优点是什么？

**参考答案：** 维度更低、计算更可控，可能更关注任务结构。

19. 真实部署为何短 horizon？

**参考答案：** 降低模型长期漂移和错误规划风险。

20. 读 World Model 代码最先看什么？

**参考答案：** latent 定义、dynamics 输入、时间对齐、loss、rollout、planner 和闭环评估。

## 13. 掌握度检查（附答案）

1. World Model 与普通 encoder 的区别是什么？

**答案：** World Model 不只编码当前输入，还要预测动作影响下的未来。

2. latent 为什么重要？

**答案：** 它是预测、规划和控制的中间状态。

3. dynamics model 的核心关系是什么？

**答案：** 当前状态与动作决定下一状态分布或预测。

4. reward/terminal head 为什么需要？

**答案：** 评估未来收益并判断任务结束。

5. 为什么 one-step loss 不够？

**答案：** 递归 rollout 会积累误差，闭环更依赖多步质量。

6. 为什么模型会进入 OOD 区域？

**答案：** planner 选择数据未覆盖动作或模型自身预测逐步漂移。

7. uncertainty 如何帮助规划？

**答案：** 对不确定未来进行惩罚、拒绝或缩短规划。

8. MPC 的闭环优势是什么？

**答案：** 每一步用真实新观测修正模型误差。

9. 为什么 latent 需要控制相关信息？

**答案：** 规划要比较动作后果，只有外观信息不够。

10. World Model 如何连接 JEPA？

**答案：** JEPA 可用未来表示预测作为 World Model 的训练目标之一。

11. World Model 如何连接 VLA？

**答案：** VLA 可产生动作条件，World Model 预测执行结果或未来状态。

12. 为什么要区分 predicted latent 和 target latent？

**答案：** 它们承担模型预测与监督目标不同角色，梯度路径可能不同。

13. 何时查 [[Video_World_Model]]？

**答案：** 需要理解视频帧、时间 token 和视觉未来预测时。

14. 何时查 [[Latent_Planning]]？

**答案：** 需要在 latent 空间搜索候选未来时。

15. 读 World Model research code 最先确认哪五项？

**答案：** state/latent、action time、target、rollout horizon、real closed-loop evaluation。
