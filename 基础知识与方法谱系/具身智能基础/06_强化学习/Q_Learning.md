# Q-Learning：通过动作价值学习选择行为

Q-Learning 是经典的离策略（Off-Policy）强化学习算法。它直接学习 Q(s,a)，估计在状态 s 采取动作 a 后的长期回报，再逐渐偏向价值更高的动作。

本文重点负责 Q 表、TD 更新、epsilon-greedy、离策略与 DQN 的连接；价值概念见 [[Value_Function]]，完整环境定义见 [[MDP]]。

## 0. 学习目标

学完以后能够：

- 解释 Q(s,a) 为什么可以用于选动作；
- 读懂 Q-Learning 的 Bellman optimality update；
- 区分 exploration 与 exploitation；
- 理解 epsilon-greedy、replay buffer 和 target network；
- 看懂从 Q 表到 DQN 的结构变化；
- 调试 target、done mask、动作索引和过估计问题。

## 1. 为什么需要 Q-Learning？

如果只知道 V(s)，还需要额外比较不同动作。Q(s,a) 直接回答“在这个状态选这个动作，长期有多好”。

```text
状态 s
↓
分别估计 Q(s,a_1), Q(s,a_2), ...
↓
选择较高 Q 的动作
↓
得到 reward 和 next_state
↓
更新当前 Q
```

Q-Learning 不要求知道环境的转移模型，因此属于 model-free 方法。

## 2. Q 表与 Q 函数

### 2.1 Q-table

离散状态和动作很少时，可以用表格保存每一对 (s,a) 的价值。表格方法直观，但状态空间稍大就无法存储。

### 2.2 Q Function

对于连续或高维状态，使用神经网络近似：

$$
Q_\theta(s,a)
$$

DQN 通常输入 state，输出所有离散动作的 Q 值向量。

```python
q_net = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, num_actions),
)

q_values = q_net(obs)
print(q_values.shape)  # [batch, num_actions]
```

动作索引 a 用 gather 取出对应 Q 值。

## 3. Q-Learning 更新公式

最优 Bellman target：

$$
y=r+\gamma \max_{a'} Q(s',a')
$$

更新：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha[y-Q(s,a)]
$$

若 episode 在当前 transition 后终止，则：

$$
y=r
$$

这里 max 假设下一状态会选择最优动作，这也是 Q-Learning 离策略的来源之一。

## 4. Exploration 与 Exploitation

- exploitation：选择当前 Q 最大的动作；
- exploration：尝试其他动作，发现未知高回报路径。

epsilon-greedy：

- 以 epsilon 概率随机动作；
- 以 1-epsilon 概率选最大 Q；
- 训练中通常逐渐降低 epsilon；
- 评估时通常关闭随机探索或固定策略。

探索太少会早早陷入次优行为，探索太多会训练不稳定。

## 5. DQN 的训练步骤

```python
q_values = online_net(states)
chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
    target = rewards + gamma * (1 - dones.float()) * next_q

loss = loss_fn(chosen_q, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

逐项理解：

- online_net 估计当前 Q；
- gather 取实际执行动作的 Q；
- target_net 提供较稳定的下一状态目标；
- done mask 在终止时去掉未来价值；
- 只优化 chosen_q 与 target 的误差。

## 6. Replay Buffer 与 Target Network

### 6.1 Replay Buffer

把 transition 存起来，随机采样 batch，打破连续样本相关性，提高数据复用。

### 6.2 Target Network

target network 是 online network 的延迟副本，周期性或软更新，避免目标随着当前网络每一步剧烈变化。

DQN 还可能使用 Double DQN、Prioritized Replay 等改进；目前先掌握它们要解决过估计和采样效率问题。

## 7. 在具身智能中的适用性

经典 DQN 更适合离散动作。机器人连续控制通常更常见 Actor-Critic 或连续动作算法。Q-Learning 仍可用于：

- 离散导航动作；
- 高层技能选择；
- 选取离散抓取候选；
- option/skill planning；
- World Model 中的离散规划动作。

## 8. 常见错误与 Debug

### 8.1 gather 动作索引错

actions 必须是合法离散索引，shape 通常要变成 [B,1] 后 gather。检查 dtype、范围和 batch 对齐。

### 8.2 target 没有 detach/no_grad

target 不应通过 online 更新目标本身，使用 no_grad 或 detach。

### 8.3 done mask 处理错

终止 transition 不应 bootstrap；注意 terminated/truncated 的定义。

### 8.4 epsilon 评估时仍很大

评估行为会随机，成功率波动且无法公平比较。分开 train_action 和 eval_action。

### 8.5 Q 值发散

检查 reward 尺度、gamma、学习率、target network、数据重复和梯度裁剪。

### 8.6 连续动作硬套 DQN

DQN 输出有限离散动作，不能直接选择连续向量。需要离散化但动作空间会快速膨胀，或使用连续动作算法。

## 9. 容易混淆

| 概念 | 含义 | 区别 |
|---|---|---|
| Q(s,a) | 当前动作价值 | V 不指定动作 |
| online net | 当前学习网络 | target net 是延迟副本 |
| epsilon | 随机探索概率 | 学习率不是它 |
| replay | 历史数据随机采样 | on-policy 当前批次 |
| done mask | 是否屏蔽未来价值 | action mask |
| max next Q | 最优目标假设 | 实际 next reward |
| DQN | 神经 Q 近似 | 原始 Q-table |

## 10. 与其他笔记的关系

前置：[[MDP]]、[[Value_Function]]、[[损失函数]]、[[优化器]]、[[PyTorch]]。

相关：[[Actor_Critic]]、[[Model_Based_RL]]、[[Offline_RL]]、[[模型评估]]。

本篇不展开连续动作 Q 算法和所有 DQN 变体；当前重点是基本更新和代码阅读。

## 11. 知识地图

```text
state
↓
online Q network
↓
epsilon-greedy action
↓
transition / replay
↓
target network
↓
Bellman target
↓
Q loss 与更新
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| q_values | [B,num_actions] |
| actions | Long、范围合法 |
| gather | 取实际动作 Q |
| target | reward + gamma × next Q |
| done | 终止屏蔽 bootstrap |
| epsilon | 训练探索、评估关闭 |
| target_net | 延迟更新 |

## 13. 练习题（题目下附参考答案）

1. Q(s,a) 表示什么？

**参考答案：** 在状态 s 采取动作 a 后继续行动的期望长期回报。

2. Q-Learning 为什么是 model-free？

**参考答案：** 它不需要显式学习或知道环境转移模型，直接从交互更新 Q。

3. epsilon-greedy 如何平衡什么？

**参考答案：** 在探索未知动作和利用当前高价值动作之间平衡。

4. DQN 输出 [B,num_actions] 代表什么？

**参考答案：** 每个 batch 样本对所有离散动作的 Q 值。

5. 为什么要 gather？

**参考答案：** 从所有动作 Q 值中取出实际执行动作对应的 Q。

6. Bellman optimality target 的核心是什么？

**参考答案：** 当前 reward 加折扣后的下一状态最大 Q 值。

7. 终止状态的 target 为什么通常不含 next Q？

**参考答案：** 终止后没有未来环境回报，继续 bootstrap 会污染目标。

8. target network 解决什么问题？

**参考答案：** 让训练目标变化更慢、更稳定，减少 online network 自我追逐。

9. replay buffer 的作用是什么？

**参考答案：** 随机复用历史 transition，打破连续样本相关并提高数据效率。

10. 连续机器人动作能直接用 DQN 吗？

**参考答案：** 不能直接，DQN 输出有限离散动作；通常使用连续动作算法或谨慎离散化。

11. actions dtype 为什么重要？

**参考答案：** gather 的索引通常要求 Long，且索引必须在动作范围内。

12. Q 值发散时检查什么？

**参考答案：** reward、gamma、学习率、done mask、target 更新、梯度和数据分布。

13. 为什么训练时需要探索？

**参考答案：** 只利用当前估计可能永远不尝试未知但更好的动作。

14. 评估时 epsilon 应怎样处理？

**参考答案：** 通常关闭随机探索或固定很小的探索率，保证评估可重复。

15. Q-Learning 如何选择动作？

**参考答案：** 评估阶段通常选择 Q 值最大的合法动作。

16. Double DQN 主要针对什么？

**参考答案：** 减轻使用同一网络取 max 和评估导致的 Q 值过估计。

17. 机器人导航为什么适合离散 Q？

**参考答案：** 上/下/左/右或离散技能等动作可以自然表示为有限动作集合。

18. Offline 数据训练 DQN 有什么风险？

**参考答案：** 网络可能高估数据中没出现的动作，策略选择分布外行为。

19. World Model 中 Q 值可以做什么？

**参考答案：** 对 imagined rollout 的离散动作和未来状态进行价值比较。

20. 读 DQN 代码最先看什么？

**参考答案：** Q 输出 shape、动作 gather、target 构造、done mask、target network 更新和 epsilon。

## 13. 掌握度检查（附答案）

1. Q 与 V 的区别是什么？

**答案：** Q 指定状态和动作，V 只指定状态并对策略动作求期望。

2. Q-Learning 的目标是什么？

**答案：** 学习最优动作价值，使 Q 能指导长期回报最大的动作选择。

3. epsilon-greedy 的两个分支是什么？

**答案：** 以 epsilon 随机探索，以 1-epsilon 选择当前最大 Q。

4. DQN 为什么输出每个动作一个 Q？

**答案：** 离散动作有限，可以一次计算所有动作再选择或取实际动作。

5. replay buffer 为什么打破相关性？

**答案：** 随机抽取不同时间和 episode 的 transition，而非连续相邻样本。

6. target network 为什么稳定？

**答案：** 目标网络延迟更新，不会随 online 网络每次更新同时剧烈变化。

7. gather 在 DQN 中做什么？

**答案：** 取执行动作对应的预测 Q 值。

8. done mask 如何影响 target？

**答案：** 终止时将未来 Q 项置零。

9. 为什么 target 常用 no_grad？

**答案：** 阻断目标分支梯度，避免改变目标计算本身。

10. Q 值发散可能有哪些原因？

**答案：** reward/gamma 过大、学习率过高、终止处理错、目标不稳定或数据异常。

11. DQN 适合什么动作空间？

**答案：** 有限离散动作空间。

12. DQN 在连续控制中的限制是什么？

**答案：** 离散化会导致动作组合爆炸、精度不足和规划困难。

13. 什么时候查 [[Value_Function]]？

**答案：** 需要理解 Bellman、TD target、Q/V/A 关系时。

14. 什么时候查 [[Actor_Critic]]？

**答案：** 需要从价值估计更新连续动作策略时。

15. 读 Q-learning research code 最先确认哪五项？

**答案：** 动作空间、Q 输出、gather、target/done、replay/target network。
