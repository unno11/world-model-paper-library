# Value Function：估计未来回报的价值

价值函数（Value Function）回答一个问题：从当前状态或状态—动作对出发，按照某种策略继续行动，未来大概能获得多少回报。

它把很长的决策后果压缩成一个可学习的数字，是 Q-Learning、Actor-Critic、规划和 World Model 中的重要桥梁。MDP 基础见 [[MDP]]，优化器见 [[优化器]]。

## 0. 学习目标

学完以后能够：

- 区分 state value V(s) 与 action value Q(s,a)；
- 解释 Bellman expectation 和 Bellman optimality 的直觉；
- 读懂 TD target、TD error 和 bootstrap；
- 理解 value network 如何训练；
- 判断 value 估计偏差、终止处理和 target detach；
- 将价值函数连接到策略评估、控制和规划。

## 1. 为什么需要价值函数？

直接等待一个 episode 结束才能知道总回报，学习信号很晚。价值函数提供当前状态“未来有多好”的预测：

```text
当前状态 s_t
↓
价值函数 V(s_t)
↓
预测未来累计回报
↓
帮助评价策略或选择动作
```

如果知道每个动作的 Q(s,a)，就可以比较动作并选择价值更高的动作。

## 2. 核心概念

### 2.1 State Value V(s)

按照策略 π 从状态 s 开始的期望回报：

$$
V^\pi(s)=E_\pi[G_t|s_t=s]
$$

它回答：在这个状态下继续遵循策略，平均能得到多少。

### 2.2 Action Value Q(s,a)

先在状态 s 执行动作 a，再继续遵循策略：

$$
Q^\pi(s,a)=E_\pi[G_t|s_t=s,a_t=a]
$$

它回答：当前选择某动作后，这个动作长期有多好。

### 2.3 Advantage

优势函数：

$$
A(s,a)=Q(s,a)-V(s)
$$

它表示动作相对当前状态平均水平好多少。Actor-Critic 常用 advantage 降低策略梯度方差。

### 2.4 Bellman Equation

价值可以拆成即时奖励加下一状态价值：

$$
V^\pi(s_t)=E[r_{t+1}+\gamma V^\pi(s_{t+1})]
$$

Q 也有类似递推。模型不必等待完整未来，可以用下一步估计更新当前估计，这叫 bootstrap。

## 3. TD Learning

TD（Temporal-Difference）学习使用一步目标：

$$
y_t=r_{t+1}+\gamma V(s_{t+1})
$$

TD error：

$$
\delta_t=y_t-V(s_t)
$$

如果终止，通常不应把未来价值加进去：

$$
y_t=r_{t+1}
$$

代码中要根据 terminated/truncated 的定义处理。

## 4. 从零读 value network

```python
class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

value_net = ValueNet(obs_dim=12)
v = value_net(obs)
print(v.shape)  # [batch]
```

最后输出 1 代表每个状态一个标量价值。squeeze(-1) 去掉最后的长度 1，但不能把 batch 维错误删掉。

## 5. Monte Carlo 与 TD 的区别

| 方法 | 目标 | 优点 | 风险 |
|---|---|---|---|
| Monte Carlo | 完整 episode return | 目标较直接 | 等待久、方差大 |
| TD(0) | reward + 下一状态估计 | 更新快、方差低 | 有 bootstrap 偏差 |
| n-step | 多步 reward + bootstrap | 折中 | 需要选 n |
| GAE | 平滑 advantage | 常用于 policy gradient | 参数更复杂 |

## 6. 价值函数用于策略评估和控制

- policy evaluation：估计给定策略的 V/Q；
- policy improvement：倾向选择价值更好的动作；
- actor-critic：critic 估计价值，actor 更新策略；
- planning：搜索或 rollout 时比较未来价值；
- offline RL：从固定数据估计价值，但分布外动作可能不可靠。

## 7. 在具身智能和 World Model 中

价值函数可表示：

- 机器人从当前状态完成任务的可能回报；
- 当前动作对未来成功的影响；
- 规划候选轨迹的终点价值；
- 模型 rollout 中的终止/成功预测。

但真实机器人上的价值估计受视觉误差、稀疏奖励、分布偏移和安全约束影响。不能把数值高直接等同于真实成功。

## 8. 常见错误与 Debug

### 8.1 把 Q 和 V 混淆

V 只接状态，Q 接状态和动作；检查输入和输出接口。

### 8.2 终止状态仍 bootstrap

成功或失败终止时，目标通常不应继续加未来价值。保留 terminated 标志并显式计算 mask。

### 8.3 target 没有 detach

训练 critic 时，target 中的下一价值常应 stop-gradient，否则目标网络本身也被反向传播，改变优化含义。

### 8.4 value loss 很低但策略差

价值预测可能在训练数据上好却泛化差，或 reward 定义、动作分布和评估不匹配。

### 8.5 数值尺度太大

奖励尺度、gamma、长 horizon 会导致价值数值很大。检查 reward 分布、return 范围和网络输出。

## 9. 容易混淆

| 概念 | 含义 | 区别 |
|---|---|---|
| V(s) | 状态价值 | 不指定当前动作 |
| Q(s,a) | 状态—动作价值 | 指定当前动作 |
| A(s,a) | 相对优势 | Q 减 V |
| return | 实际/采样回报 | 训练目标样本 |
| TD target | bootstrap 目标 | reward 加估计未来 |
| TD error | target 与当前估计差 | 更新信号 |
| critic | 学 value 的模型 | 不一定输出动作 |

## 10. 与其他笔记的关系

前置：[[MDP]]、[[概率统计]]、[[损失函数]]、[[优化器]]、[[PyTorch]]。

后续：[[Q_Learning]]、[[Policy_Gradient]]、[[Actor_Critic]]、[[Model_Based_RL]]、[[Offline_RL]]。

本篇不展开特定算法的完整训练伪代码，重点是价值概念和 TD 目标。

## 11. 知识地图

```text
state / action
↓
V(s)、Q(s,a)
↓
return 或 Bellman target
↓
TD error
↓
critic 更新
↓
策略评估、改进、规划
```

## 12. 快速查询表

| 问题 | 判断 |
|---|---|
| 只输入 state | 可能是 V |
| 输入 state+action | 可能是 Q |
| target | reward + gamma × next value |
| done | 终止时屏蔽 bootstrap |
| advantage | Q - V |
| critic loss | value prediction 与 target 的误差 |

## 13. 练习题（题目下附参考答案）

1. V(s) 回答什么问题？

**参考答案：** 从状态 s 出发按某策略继续行动，未来期望能得到多少回报。

2. Q(s,a) 与 V(s) 的区别是什么？

**参考答案：** Q 指定当前动作，V 只指定状态并平均考虑策略动作。

3. advantage 表示什么？

**参考答案：** 某动作相对当前状态平均价值的好坏。

4. TD target 的基本形式是什么？

**参考答案：** reward 加折扣后的下一状态价值，终止时通常不加未来项。

5. bootstrap 是什么？

**参考答案：** 使用当前已有的下一状态估计构造目标，而不是等待完整真实回报。

6. ValueNet 输出 [B,1] 为什么常 squeeze(-1)？

**参考答案：** 每个样本只有一个标量价值，去掉最后的 1 得到 [B]，但要保留 batch 维。

7. Monte Carlo 与 TD 的主要差异是什么？

**参考答案：** Monte Carlo 等完整回报，TD 使用下一状态估计快速更新。

8. 终止状态为什么不能随意 bootstrap？

**参考答案：** 终止后没有继续的未来状态，加入未来估计会污染目标。

9. critic 的职责是什么？

**参考答案：** 估计状态或状态—动作的价值，为策略更新或动作选择提供信号。

10. value loss 很低但策略失败可能为什么？

**参考答案：** 数据分布、reward、评估任务或价值泛化不匹配，低训练 loss 不代表闭环成功。

11. 如何判断一个网络输出 V 还是 Q？

**参考答案：** 看输入是否包含 action、输出语义和训练目标。

12. target detach 的目的是什么？

**参考答案：** 防止梯度通过 bootstrap 目标回传到用于构造目标的价值估计。

13. 稀疏奖励会怎样影响价值学习？

**参考答案：** 学习信号延迟，价值估计更难，可能需要更长轨迹、奖励设计或更合适算法。

14. Q 值过大时检查什么？

**参考答案：** reward scale、gamma、终止 mask、bootstrapping、学习率和数据是否重复。

15. 机器人规划为什么需要价值函数？

**参考答案：** 评价当前或候选未来状态的完成潜力，帮助选择更有希望的动作/轨迹。

16. Offline RL 中价值函数有什么风险？

**参考答案：** 对数据未覆盖的动作外推可能严重错误，导致策略选择不可靠动作。

17. GAE 解决什么问题？

**参考答案：** 在 advantage 估计的偏差与方差之间折中，常用于 Actor-Critic。

18. 为什么应画 value prediction 与 return？

**参考答案：** 检查尺度、偏差、终止处理和不同状态区域的拟合情况。

19. World Model 里的 value head 可能预测什么？

**参考答案：** 未来回报、成功概率、终止概率或规划评价分数。

20. 读 critic 代码最先看什么？

**参考答案：** 输入 state/action、输出 shape、target 公式、done mask、detach 和 loss。

## 13. 掌握度检查（附答案）

1. V 与 Q 的定义分别是什么？

**答案：** V 是状态价值，Q 是状态—动作价值，都表示未来期望回报。

2. A(s,a) 是什么？

**答案：** Q(s,a)-V(s)，表示动作相对于状态平均水平的优势。

3. Bellman 递推的核心是什么？

**答案：** 当前价值等于即时 reward 加折扣的未来价值。

4. TD learning 为什么叫时序差分？

**答案：** 用相邻时间步价值估计之间的差异更新，而不必等待完整 episode。

5. Monte Carlo 的优缺点是什么？

**答案：** 目标真实直接但等待长、方差大。

6. TD 的优缺点是什么？

**答案：** 更新快、方差较低，但引入 bootstrap 偏差。

7. terminated 如何影响 TD target？

**答案：** 自然终止时通常屏蔽未来 value 项。

8. critic 与 actor 的关系是什么？

**答案：** critic 评估价值，actor 根据价值/advantage 改进策略。

9. 为什么 target 常 detach？

**答案：** 让目标在当前 critic 更新中被视作固定目标，避免错误梯度路径。

10. value 网络输出标量意味着什么？

**答案：** 对每个状态或状态—动作输入给出一个价值估计。

11. 价值函数在规划中有什么作用？

**答案：** 评估候选未来的长期收益，补充有限 rollout。

12. Offline RL 的 value 外推问题是什么？

**答案：** 数据没有覆盖的动作价值可能被高估，策略会选择未见过的危险动作。

13. 什么时候查 [[MDP]]？

**答案：** 需要重新确认状态、动作、奖励、终止和时间关系时。

14. 什么时候查 [[Actor_Critic]]？

**答案：** 需要理解 value 如何作为 policy gradient 的 baseline 或 advantage 时。

15. 读 value research code 最先确认哪五项？

**答案：** 输入语义、target、gamma、终止 mask、detach/评估方式。
