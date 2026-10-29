# MDP：用状态、动作和奖励描述决策问题

马尔可夫决策过程（Markov Decision Process，MDP）是强化学习的基本语言。它把“智能体在环境中观察状态、选择动作、得到奖励并进入下一状态”的过程形式化。

本文负责强化学习的共同地基：状态、动作、奖励、转移、策略、折扣、回合和 return。具体价值学习见 [[Value_Function]]，Q-Learning 见 [[Q_Learning]]，策略优化见 [[Policy_Gradient]]。

## 0. 学习目标

学完以后能够：

- 解释 agent、environment、state、action、reward 和 transition；
- 区分 state、observation、history 和 Markov property；
- 读懂 trajectory、episode、return 和 discount；
- 判断 reward 与任务目标是否一致；
- 将 MDP 映射到机器人、模仿学习和 World Model；
- 排查时间错位、奖励错误和终止条件错误。

## 1. 为什么需要 MDP？

普通监督学习给定输入和标签，样本之间通常可独立处理。决策任务不同：

```text
当前状态 s_t
↓
策略 π 选择动作 a_t
↓
环境执行动作
↓
得到奖励 r_{t+1}
↓
进入下一状态 s_{t+1}
↓
继续决策
```

当前动作会改变未来状态，未来奖励又依赖后续动作。因此不能只预测一个静态标签。

## 2. 核心概念

### 2.1 Agent 与 Environment

- agent：做决策的学习系统；
- environment：接收动作、推进世界并返回结果；
- state：描述决策所需环境信息；
- action：agent 可执行的选择；
- reward：环境给出的即时反馈。

机器人中 agent 可能是 policy，environment 可能是仿真器或真实机器人。

### 2.2 State、Observation 与 History

state 是理想上足以决定未来的环境描述；observation 是 agent 实际看到的信息；history 是过去的观测和动作。

如果 observation 不包含完整 state，就叫部分可观测问题，常用 RNN/LSTM 或 Transformer 记忆历史，见 [[RNN_LSTM]]、[[Transformer]]。

### 2.3 Markov Property

马尔可夫性质直觉是：给定当前 state 后，未来与更早历史无关：

$$
P(s_{t+1}|s_t,a_t)
$$

如果当前观测不够，真实系统可能不满足这个简化条件，需要扩展状态或使用历史。

### 2.4 Transition

transition 是：

$$
(s_t,a_t,r_{t+1},s_{t+1},done)
$$

其中 done 表示 episode 是否结束。代码中还可能有 truncated，表示因为时间上限或外部限制而截断，不一定是任务自然终止。

### 2.5 Policy

策略（Policy）是根据状态选择动作的规则：

- 确定性策略：a=π(s)；
- 随机策略：a ~ π(a|s)。

代码里 policy 可能是神经网络输出动作、动作 logits 或动作分布。

### 2.6 Return 与 Discount

从时间 t 开始的折扣回报：

$$
G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots
$$

gamma 在 0 和 1 之间。gamma 越小越重视近处奖励，越大越重视长期结果。

## 3. MDP 的整体工作原理

```text
初始化环境
↓
reset → s_0
↓
policy(s_t) → a_t
↓
env.step(a_t)
↓
(r_{t+1}, s_{t+1}, terminated, truncated)
↓
记录 transition
↓
结束？否：继续
↓
episode return / update
```

强化学习算法的差别，主要在于如何利用这些 transition 更新 policy、value 或模型。

## 4. 读懂环境接口

```python
obs, info = env.reset()
done = False
episode_return = 0.0

while not done:
    action = policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_return += reward
    obs = next_obs
```

要检查：

- reset 返回的 observation shape；
- action 是否在合法范围；
- reward 的符号和尺度；
- terminated 与 truncated 的含义；
- next_obs 是否与 action 时间对齐。

## 5. Reward Design

奖励是优化目标的一部分，不只是日志数字。奖励过稀疏，学习可能没有方向；奖励过密集，可能鼓励捷径。

机器人例子：

- 到目标更近：正奖励；
- 成功抓取：大正奖励；
- 碰撞：负奖励；
- 每步时间成本：小负奖励。

Reward shaping 可能改变最优策略，必须验证它是否仍对应真实任务目标。

## 6. Episode、Trajectory 与 Replay

- episode：从 reset 到终止的一段交互；
- trajectory：按时间排列的一串 transition；
- replay buffer：保存历史 transition，供离策略算法反复采样；
- on-policy batch：由当前策略新采集的数据。

这些数据结构决定算法能否重复使用旧数据，见 [[Q_Learning]]、[[Offline_RL]]。

## 7. 在 AI 与具身智能中的使用

```text
机器人观测 o_t
↓
状态编码器 / policy
↓
动作 a_t
↓
物理环境
↓
新观测、奖励、终止
↓
策略更新或轨迹记录
```

在 World Model 中，模型试图学习：

$$
p(s_{t+1},r_{t+1}|s_t,a_t)
$$

它可以在真实交互较少时进行 imagined rollout，但模型误差会随时间积累，见 [[Model_Based_RL]]。

## 8. 常见错误与 Debug

### 8.1 reward 时间错位

确认 reward 是 action_t 执行后得到的 r_{t+1}，不要把它错配到下一动作。

### 8.2 terminated 与 truncated 混淆

自然成功/失败终止和时间上限截断对 bootstrap 处理可能不同。记录原始标志，不要只保存 done。

### 8.3 observation 不是真实 state

部分可观测时，策略可能需要历史。不要因为变量叫 state 就假设它满足 Markov property。

### 8.4 action 范围错误

策略输出可能是 logits、未限制实数或 [-1,1]，环境却要求特定范围。检查缩放、clip 和单位。

### 8.5 reward scale 过大或符号反了

打印 reward 分布、episode return 和成功率，用极小环境手工执行一条轨迹。

## 9. 容易混淆

| 概念 | 含义 | 不等于 |
|---|---|---|
| state | 决策所需环境描述 | 一定是可见图像 |
| observation | agent 实际输入 | 完整 state |
| action | 当前选择 | 下一状态 |
| reward | 即时反馈 | return |
| return | 累积折扣奖励 | 单步 reward |
| episode | 一段完整交互 | 一个 batch |
| terminated | 自然终止 | 必然成功 |
| truncated | 外部截断 | 失败标签 |

## 10. 与其他笔记的关系

前置：[[概率统计]]、[[监督学习]]、[[PyTorch]]、[[模型评估]]。

后续：[[Value_Function]]、[[Q_Learning]]、[[Policy_Gradient]]、[[Actor_Critic]]、[[Model_Based_RL]]、[[Offline_RL]]、[[Imitation_Learning]]。

本篇不展开具体算法更新公式；它们分别由其他笔记负责。

## 11. 知识地图

```text
MDP
├─ state / observation
├─ action
├─ transition
├─ reward
├─ policy
├─ return / discount
└─ episode / trajectory
↓
value、Q、policy 或 model 学习
↓
交互与评估
```

## 12. 快速查询表

| 看到的字段 | 先问什么 |
|---|---|
| obs/state | 是否完整、shape 是什么 |
| action | 范围、单位、离散/连续 |
| reward | 时刻、符号、尺度 |
| next_obs | 是否由当前 action 得到 |
| terminated | 自然终止条件 |
| truncated | 时间或外部截断 |
| gamma | 长期回报权重 |

## 13. 练习题（题目下附参考答案）

1. MDP 用来描述什么？

**参考答案：** 描述智能体在状态中选择动作、获得奖励并转移到下一状态的序贯决策过程。

2. agent 和 environment 分别做什么？

**参考答案：** agent 选择动作，environment 执行动作并返回新状态、奖励和终止信息。

3. reward 与 return 的区别是什么？

**参考答案：** reward 是单步反馈，return 是从当前时刻开始的折扣累计奖励。

4. observation 一定等于 state 吗？

**参考答案：** 不一定。部分可观测环境中 observation 只是 state 的不完整信息。

5. policy 是什么？

**参考答案：** 根据状态选择动作的规则或动作分布。

6. gamma 接近 0 会怎样？

**参考答案：** 更重视即时奖励，较少关注远期奖励。

7. 一条 transition 包含哪些核心字段？

**参考答案：** state/observation、action、reward、next state/observation，以及终止信息。

8. terminated 与 truncated 为什么要区分？

**参考答案：** 一个是环境自然结束，一个是外部时间限制等截断，后续价值 bootstrap 处理可能不同。

9. reward shaping 有什么风险？

**参考答案：** 可能鼓励与真实任务不一致的捷径，改变最优策略。

10. 部分可观测时可以怎么办？

**参考答案：** 使用历史堆叠、RNN/LSTM、Transformer 或学习状态估计器。

11. action 输出范围错误如何排查？

**参考答案：** 检查 policy 输出含义、tanh/scale/clip、环境 action_space 和单位。

12. 如果 reward 符号反了，会出现什么？

**参考答案：** agent 可能学到主动追求失败或碰撞，episode return 和行为方向会异常。

13. episode、trajectory、replay buffer 区别是什么？

**参考答案：** episode 是一段交互，trajectory 是按时间排列的 transition，buffer 保存多段历史以供采样。

14. 为什么不能把所有 done 都当自然终止？

**参考答案：** 时间上限截断不一定表示失败或成功，可能影响目标价值估计。

15. 机器人 policy 的 observation 可能包含什么？

**参考答案：** 图像、深度、关节状态、末端位姿、目标和历史信息。

16. World Model 试图建模什么？

**参考答案：** 给定当前状态和动作时，下一状态与奖励的条件分布或预测。

17. 如何验证环境 reward？

**参考答案：** 手工执行成功/失败轨迹，打印每步 reward、return、终止标志和可视化状态。

18. 为什么离线数据中要保存时间戳？

**参考答案：** 对齐传感器、动作、reward 和不同帧率，避免 transition 错位。

19. 强化学习和监督学习的标签有什么不同？

**参考答案：** 强化学习的目标由未来回报和策略影响，通常不是静态人工标签。

20. 读 RL repository 最先看什么？

**参考答案：** 环境接口、observation/action space、reward、终止规则和数据记录方式。

## 13. 掌握度检查（附答案）

1. MDP 的五个核心要素是什么？

**答案：** 状态、动作、转移、奖励和策略/决策过程；常用完整描述还包括折扣和终止。

2. Markov property 的直觉是什么？

**答案：** 当前状态包含预测未来所需的信息，过去历史不再额外提供必要信息。

3. observation 与 state 的区别是什么？

**答案：** observation 是 agent 看到的输入，state 是理论上足以描述环境的状态。

4. reward 与 return 的区别是什么？

**答案：** reward 是即时反馈，return 是未来奖励的折扣和。

5. policy 是什么？

**答案：** 状态到动作或动作分布的映射。

6. 为什么需要 gamma？

**答案：** 控制近期开支与长期回报的权衡，并帮助定义有限/无限期回报。

7. terminated 和 truncated 的区别是什么？

**答案：** 前者是自然结束，后者是外部条件截断。

8. 为什么 action 影响学习目标？

**答案：** action 改变未来状态和奖励，决策与结果是闭环关系。

9. reward shaping 为什么需要谨慎？

**答案：** 不合适的奖励会使策略优化错误目标。

10. 什么是 trajectory？

**答案：** 按时间排列的一串状态、动作、奖励和后继状态。

11. replay buffer 的作用是什么？

**答案：** 保存过去交互数据，供离策略算法重复采样。

12. 部分可观测如何处理？

**答案：** 让输入包含历史，或用记忆模型推断隐藏状态。

13. MDP 如何连接 World Model？

**答案：** World Model 学习 MDP 中的转移和奖励动态。

14. 何时查 [[Value_Function]]？

**答案：** 需要理解状态价值、动作价值和未来回报估计时。

15. 读环境代码最先检查哪五项？

**答案：** observation、action space、reward 时间、terminated/truncated、reset/step shape。
