# Actor-Critic：让策略和价值互相协作

Actor-Critic 同时学习两个部分：

- Actor：策略，决定做什么；
- Critic：价值函数，评价当前状态或动作有多好。

Critic 为 Actor 提供较低方差的 advantage 或 TD error，使策略梯度比纯 REINFORCE 更稳定。前置概念见 [[MDP]]、[[Value_Function]]、[[Policy_Gradient]]。

## 0. 学习目标

学完以后能够：

- 解释 actor 与 critic 的职责；
- 理解 TD error 如何作为策略更新信号；
- 读懂 actor loss、critic loss 和 entropy；
- 区分共享 encoder 与独立网络；
- 理解 on-policy actor-critic 的 rollout 和更新；
- 调试 value target、advantage、终止 mask 和梯度路径。

## 1. 为什么需要 Actor-Critic？

纯策略梯度等待完整 return，方差较大；只学习价值又没有直接的策略输出。Actor-Critic 让两者分工：

```text
state
├─ Actor → action distribution → action
└─ Critic → value estimate
                    ↓
             advantage / TD error
                    ↓
             更新 Actor
```

Actor 做决策，Critic 提供评价和学习信号。

## 2. 核心概念

### 2.1 Actor

Actor 是策略 πθ(a|s)。离散动作输出 logits，连续动作输出分布参数。

### 2.2 Critic

Critic 通常估计 Vφ(s)。也有 Q critic，但输入输出和 target 不同，不能只看变量名。

### 2.3 TD Error

一步 TD error：

$$
\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

终止时去掉下一状态项。δ 可以作为 advantage 的简单估计。

### 2.4 GAE

GAE 用一串 TD error 加权，折中 bias 与 variance。常见超参数 gamma 和 lambda 共同决定未来信息传播范围。

## 3. Actor-Critic 的整体流程

```text
用当前 policy 与环境交互
↓
记录 s,a,r,done,log_prob,V(s)
↓
计算 next value、TD error、advantage、return
↓
actor_loss = -log_prob × advantage
critic_loss = value 与 return 的误差
entropy bonus
↓
合并 loss 或分别更新
↓
开始下一批 rollout
```

## 4. 一个简化网络

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.actor(h), self.critic(h).squeeze(-1)
```

共享 encoder 可以提高效率，但 actor 和 critic 的目标可能冲突；有些研究代码使用完全独立的网络。

## 5. 损失与梯度

常见形式：

$$
L_{actor}=-E[\log\pi(a_t|s_t)A_t]
$$

$$
L_{critic}=E[(V(s_t)-R_t)^2]
$$

$$
L=L_{actor}+c_vL_{critic}-c_eH
$$

critic target/advantage 常需 detach，避免 actor loss 反向改变 critic 的目标路径。具体实现要看设计。

## 6. Rollout 与 Batch

on-policy 算法通常先让多个环境并行运行固定步数，再统一计算 advantage。需要保存：

- observation；
- action；
- reward；
- done；
- log_prob；
- value；
- next value 或 bootstrap value。

时间和环境维度可能是 [T,N_env]，送入网络前常 reshape 为 [T×N_env,...]，但计算 GAE 时必须保留时间顺序。

## 7. 在具身智能中的使用

Actor-Critic 常用于：

- 连续机器人控制；
- locomotion；
- manipulation；
- 视觉策略；
- 分层技能选择；
- World Model imagined rollout 的策略优化。

真实环境中还要处理安全边界、动作平滑、观测延迟、奖励稀疏和 sim-to-real 分布偏移。

## 8. 常见错误与 Debug

### 8.1 advantage 未 detach

如果 actor loss 通过 advantage 回传到 critic，可能改变算法意图。检查是否按实现 detach。

### 8.2 done mask 错

终止后不能把 next value 继续 bootstrap。区分自然终止和时间截断。

### 8.3 actor/critic loss 尺度失衡

critic loss 过大可能压制 actor，entropy 过大可能让策略过度随机。记录每项 loss 和梯度范数。

### 8.4 shared encoder 梯度冲突

共享部分同时收到 actor 与 critic 梯度，需明确这是设计还是意外。

### 8.5 GAE 时间方向错

优势应从后往前递推，done 会重置递推。打印一条很短 rollout 手算验证。

## 9. 容易混淆

| 概念 | 含义 | 区别 |
|---|---|---|
| actor | 产生策略 | 不负责主要价值估计 |
| critic | 评价状态/动作 | 不直接决定动作 |
| value | critic 输出 | 不是 reward |
| advantage | 相对基准优势 | 不是概率 |
| rollout | 新采集交互 | 不是 replay 任意历史 |
| bootstrap | 用 next value | 不等于完整 return |
| entropy bonus | 鼓励探索 | 不等于任务 reward |

## 10. 与其他笔记的关系

前置：[[MDP]]、[[Value_Function]]、[[Policy_Gradient]]、[[损失函数]]、[[优化器]]、[[PyTorch]]。

相关：[[Q_Learning]]、[[Imitation_Learning]]、[[Model_Based_RL]]、[[Offline_RL]]、[[Transformer]]。

本篇不展开 PPO 的 clip、GAE 全部推导和分布式采样细节。

## 11. 知识地图

```text
rollout
↓
actor 采样 action
+ critic 估计 value
↓
reward / done
↓
advantage / return
├─ actor loss
└─ critic loss
↓
联合更新
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| actor output | logits 或连续分布参数 |
| critic output | 通常每状态一个标量 |
| advantage | 时间方向、detach、标准化 |
| return | bootstrap 与 done |
| loss weights | actor/critic/entropy 比例 |
| shared encoder | 梯度是否符合设计 |
| rollout | 当前策略新数据 |

## 13. 练习题（题目下附参考答案）

1. Actor 和 Critic 分别做什么？

**参考答案：** Actor 选择动作，Critic 估计状态或动作的价值，为 Actor 提供评价信号。

2. 为什么需要 Critic？

**参考答案：** 用价值估计替代高方差的完整 return，提供更稳定的 advantage/TD error。

3. TD error 的基本形式是什么？

**参考答案：** reward 加折扣下一价值减去当前价值，终止时去掉未来项。

4. actor loss 为什么有负号？

**参考答案：** 代码通常执行最小化，而策略目标是最大化高 advantage 动作的 log probability。

5. critic loss 常比较什么？

**参考答案：** critic 的 value prediction 与 return/TD target。

6. advantage 表示什么？

**参考答案：** 当前动作相对状态基准价值的好坏。

7. shared encoder 有什么好处和风险？

**参考答案：** 好处是共享表示、参数少；风险是 actor 和 critic 梯度目标可能冲突。

8. GAE 为什么从后往前计算？

**参考答案：** 它依赖未来 TD error 的递推加权。

9. 终止 mask 为什么重要？

**参考答案：** 防止终止状态错误使用不存在的未来 value。

10. rollout 保存哪些信息？

**参考答案：** state、action、reward、done、log_prob、value 及必要的 next value。

11. actor loss 通过 advantage 更新 critic 合理吗？

**参考答案：** 通常需要按算法设计对 advantage detach，避免改变 critic 目标路径。

12. critic loss 过大可能怎样？

**参考答案：** 可能压制 actor 更新，导致策略学习缓慢或不稳定。

13. entropy 太大说明什么？

**参考答案：** 策略保持高随机性，可能探索过强。

14. GAE 时间错位怎么排查？

**参考答案：** 用 3～5 步 rollout 手算 reward、value、delta 和 advantage，对照代码。

15. Actor-Critic 适合连续动作吗？

**参考答案：** 适合，actor 可以输出连续动作分布参数。

16. 真实机器人使用 Actor-Critic 要注意什么？

**参考答案：** 安全约束、动作范围、延迟、奖励设计、样本成本和 sim-to-real。

17. World Model 中 Critic 可以评价什么？

**参考答案：** imagined 状态、候选动作、未来成功概率或累计回报。

18. 如何判断 critic 输出 V 还是 Q？

**参考答案：** 看是否输入 action、输出 shape 和 target 定义。

19. 为什么 rollout 不是普通 replay buffer？

**参考答案：** on-policy rollout 必须来自当前策略，旧数据通常不能无限复用。

20. 读 Actor-Critic 代码最先看什么？

**参考答案：** actor/critic 输入输出、rollout 字段、advantage/return 计算、detach、done mask 和 loss 权重。

## 13. 掌握度检查（附答案）

1. Actor-Critic 由哪两部分组成？

**答案：** Actor 策略和 Critic 价值估计器。

2. Critic 如何帮助 Actor？

**答案：** 提供 value、TD error 或 advantage，降低策略梯度方差并指示动作好坏。

3. TD error 是什么？

**答案：** 一步 bootstrap target 与当前 value 估计的差。

4. actor loss 优化什么？

**答案：** 提高高 advantage 动作的策略概率。

5. critic loss 优化什么？

**答案：** 让价值估计接近 return 或 TD target。

6. advantage 与 return 的区别是什么？

**答案：** return 是总回报目标，advantage 是相对状态基准的动作评价。

7. 为什么需要 detach？

**答案：** 阻断不希望的梯度路径，让 actor 和 critic 按设计分工更新。

8. GAE 处理什么权衡？

**答案：** advantage 估计的偏差与方差。

9. done 对 GAE 有什么影响？

**答案：** 终止时切断未来递推，防止跨 episode 传播。

10. shared encoder 的风险是什么？

**答案：** 两个目标的梯度可能冲突，导致表示或训练不稳定。

11. 为什么要记录 actor/critic/entropy 各项 loss？

**答案：** 判断哪个目标主导训练以及系数是否合理。

12. 为什么连续控制常使用 Actor-Critic？

**答案：** actor 可直接参数化连续动作分布，critic 可提供稳定评价。

13. Actor-Critic 与 Q-Learning 的主要差异是什么？

**答案：** Actor-Critic 显式学习策略和价值，Q-Learning 主要学习 Q 并从中选动作。

14. 何时查 [[Policy_Gradient]]？

**答案：** 需要理解 log_prob、return 和策略梯度目标时。

15. 读 Actor-Critic research code 最先确认哪五项？

**答案：** actor distribution、critic 输出、rollout 时间维、advantage/return、detach 和 done mask。
