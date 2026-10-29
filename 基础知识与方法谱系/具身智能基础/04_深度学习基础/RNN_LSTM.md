# RNN 与 LSTM：让模型处理时间序列

循环神经网络（Recurrent Neural Network，RNN）和长短期记忆网络（Long Short-Term Memory，LSTM）用于处理有顺序的数据。它们逐步读取序列，并维护 hidden state，把过去的信息带到当前时刻。

本文重点负责序列、时间步、hidden state、teacher forcing、padding 和 LSTM 的门；Attention 与 Transformer 见 [[Attention]]、[[Transformer]]。现代视觉语言和机器人研究中，RNN/LSTM 仍会出现在动作序列、轨迹、低频控制和历史状态编码器中。

## 0. 学习目标

学完以后能够：

- 解释序列、时间步、hidden state 和 recurrent update；
- 区分 batch、time、feature 三个维度；
- 读懂 RNN/LSTM 输入输出 shape；
- 理解 LSTM 为什么比普通 RNN 更能保留长期信息；
- 看懂 many-to-one、many-to-many 和 teacher forcing；
- 排查 padding、hidden state、shape 和梯度问题。

## 1. 为什么需要序列模型？

单帧图像只能告诉模型当前看到什么，动作控制往往还需要知道：

- 物体刚才在哪里；
- 机器人上一时刻的速度；
- 已经执行了哪些动作；
- 当前观测是否暂时被遮挡。

序列模型把多个时间步连接起来：

```text
x_1 → RNN → h_1
x_2 → RNN → h_2
x_3 → RNN → h_3
                 ↓
              prediction
```

h_t 是到时间 t 为止的历史摘要。它不是永久记忆，而是一个由模型学习如何更新的状态。

## 2. 核心概念

### 2.1 时间序列（Sequence）

序列是有顺序的一组元素。文本 token、视频帧、机器人状态和动作轨迹都可以是序列。

常见 batch-first shape：

```text
[B, T, F]
B：样本数
T：时间步数
F：每个时间步的特征数
```

例如 [32,20,128] 表示 32 条序列，每条 20 帧，每帧 128 维。

### 2.2 Hidden State

hidden state 是循环网络在当前时间步保存的中间状态。普通 RNN 的简化公式：

$$
h_t = tanh(W_xx_t + W_hh_{t-1}+b)
$$

当前状态由当前输入和上一状态共同决定。

### 2.3 Many-to-one 与 many-to-many

- many-to-one：读完整段序列，只用最后状态分类或预测一个结果；
- many-to-many：每个时间步都输出，如动作序列标注；
- one-to-many：给一个条件，逐步生成序列。

机器人策略可能使用历史观测序列预测当前动作，也可能每一步都输出动作。

### 2.4 LSTM 的直觉

普通 RNN 长序列中容易出现梯度消失或爆炸。LSTM 增加 cell state 和门控机制：

- forget gate：忘掉什么；
- input gate：写入什么；
- output gate：暴露什么；
- cell state：相对稳定地传递长期信息。

常见代码中 LSTM 返回 output 和 (h_n,c_n)。h_n 是最后隐藏状态，c_n 是最后 cell state。

## 3. LSTM 的整体流程

```text
输入 x_t
↓
读取上一时刻 h_{t-1}, c_{t-1}
↓
forget / input / output gates
↓
得到 h_t, c_t
↓
进入下一时间步
↓
输出序列或最后状态
```

它不是魔法记忆：如果数据、损失或训练方式没有提供需要长期信息的信号，LSTM 也不会自动学会有用记忆。

## 4. 从零读代码

```python
lstm = nn.LSTM(
    input_size=16,
    hidden_size=64,
    num_layers=2,
    batch_first=True,
)

x = torch.randn(32, 20, 16)
out, (h_n, c_n) = lstm(x)

print(out.shape)  # [32, 20, 64]
print(h_n.shape)  # [2, 32, 64]
print(c_n.shape)  # [2, 32, 64]
```

解释：

- input_size=16：每个时间步的特征数；
- hidden_size=64：每层隐藏状态维度；
- num_layers=2：堆叠两层 LSTM；
- batch_first=True：输入为 [B,T,F]；
- out 保存最后一层每个时间步的输出；
- h_n 保存每层最后时间步的 hidden；
- c_n 保存每层最后时间步的 cell。

many-to-one 常取 out[:, -1, :] 或 h_n 的最后一层：

```python
sequence_feature = out[:, -1, :]
prediction = head(sequence_feature)
```

但是否取最后一步，要看任务和 padding 方式。

## 5. Padding 与变长序列

真实轨迹长度可能不同，通常 padding 到同一 T。padding 值不是有效观测，不能让模型和 loss 把它当真实数据。

可使用：

- mask：标记哪些时间步有效；
- packed sequence：让 RNN 跳过 padding；
- masked loss：只在有效位置计算误差。

如果最后一步是 padding，直接取 out[:, -1, :] 会得到错误的序列表示。

## 6. Teacher Forcing 与序列预测

在自回归预测中，模型根据过去预测下一步。训练时可以把真实的上一时刻目标作为输入，这叫 teacher forcing。

优点是训练更稳定、收敛更快；问题是训练时看真实历史，推理时只能看自己的预测，可能产生 exposure bias。

```text
训练：真实 x_t → 预测 y_{t+1}
推理：模型预测的 y_t → 作为下一步输入
```

## 7. 训练与梯度问题

序列越长，计算图越长，显存和梯度传播成本越高。可使用 truncated BPTT，把长序列切成窗口，并在窗口间按设计 detach hidden state。

梯度裁剪常用于 RNN/LSTM：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

为什么裁剪有效见 [[优化器]]；loss 的序列 mask 见 [[损失函数]]。

## 8. 在 AI 与具身智能中的使用

LSTM 可作为：

- 观测历史编码器；
- 视觉特征到动作的时序策略；
- 轨迹预测器；
- 部分可观测环境中的记忆模块；
- 低频规划与高频控制之间的状态摘要器。

典型流程：

```text
图像或 proprioception 序列 [B,T,F]
↓
LSTM
↓
历史表示 [B,H]
↓
MLP policy head
↓
当前动作
```

与 World Model 的联系是：LSTM 可以维护 latent dynamics 的历史状态，但现代模型也常用 Transformer 处理更长上下文。

## 9. 常见错误与 Debug

### 9.1 维度顺序错误

batch_first=True 需要 [B,T,F]；默认设置通常是 [T,B,F]。两者数值可能都能运行但语义完全不同。

### 9.2 hidden state batch 不匹配

hidden 的 batch 维必须和输入 batch 一致。改变 batch size、最后一个不完整 batch 或跨 batch 复用 hidden 时尤其容易出错。

### 9.3 把 padding 当有效数据

必须使用长度、mask 或 packed sequence。否则最后状态和 loss 都可能被 padding 污染。

### 9.4 推理时忘记 eval 和 no_grad

Dropout、BatchNorm 或其他模块会受 train/eval 影响；推理时通常使用 model.eval() 与 no_grad()，详见 [[PyTorch]]。

### 9.5 梯度爆炸

观察梯度范数和 loss；降低学习率、规范化输入、裁剪梯度，并检查序列目标是否错位。

## 10. 容易混淆

| 概念 | 含义 | 不等于 |
|---|---|---|
| time step | 序列中的一个位置 | batch |
| hidden state | 历史摘要 | 标签 |
| cell state | LSTM 的长期传递状态 | 参数 |
| output | 每个时间步的输出 | 只有最后状态 |
| padding | 补齐长度的占位 | 真数据 |
| teacher forcing | 训练时使用真实历史 | 推理策略 |
| mask | 标记有效位置 | 改变数据 shape |

## 11. 与其他笔记的关系

前置：[[PyTorch]]、[[监督学习]]、[[损失函数]]、[[优化器]]、[[模型评估]]。

后续：[[Attention]]、[[Transformer]]、[[表征学习]]。

本篇不展开注意力的 Q/K/V 和 Transformer block；需要更长上下文时转到对应笔记。

## 12. 知识地图

```text
序列 [B,T,F]
↓
每个时间步输入
↓
更新 h、c
↓
输出所有时间步或最后有效状态
↓
任务 head
↓
masked loss
↓
反向传播与评估
```

## 13. 快速查询表

| 问题 | 检查 |
|---|---|
| 输入 shape | batch_first 与 [B,T,F] |
| 取最后状态 | 最后位置是否为有效时间步 |
| 变长序列 | length、mask、packed sequence |
| 梯度爆炸 | loss、grad norm、clip |
| 训练推理差异 | train/eval、teacher forcing |

## 14. 练习题（题目下附参考答案）

1. 序列模型为什么需要时间维？

**参考答案：** 时间维保存元素的顺序，使模型可以利用过去信息理解当前输入。

2. [32,20,16] 中三个数字是什么？

**参考答案：** 32 条序列、每条 20 个时间步、每步 16 个特征。

3. hidden state 的作用是什么？

**参考答案：** 汇总到当前时刻的历史信息，并传给下一个时间步。

4. LSTM 比普通 RNN 多了什么重要机制？

**参考答案：** cell state 和 forget/input/output gates，用于控制信息保留、写入和输出。

5. many-to-one 是什么？

**参考答案：** 输入多个时间步，最终只产生一个结果，如序列分类。

6. batch_first=True 时输入 shape 是什么？

**参考答案：** [B,T,F]。

7. LSTM output 与 h_n 有何区别？

**参考答案：** output 是最后一层每个时间步的输出，h_n 是每层最后时间步的 hidden state。

8. 为什么不能无条件取 out[:, -1, :]？

**参考答案：** 变长序列 padding 时最后位置可能不是有效数据，应取最后有效位置或使用 mask/length。

9. teacher forcing 的优点和风险是什么？

**参考答案：** 优点是训练稳定；风险是训练看真实历史、推理看自身预测，存在 exposure bias。

10. 序列任务为何使用 masked loss？

**参考答案：** 只在有效时间步计算误差，避免 padding 对训练产生错误影响。

11. RNN/LSTM 在机器人中可能处理什么？

**参考答案：** 观测历史、动作轨迹、部分可观测状态和时序策略。

12. batch size 改变后 hidden 报错，原因是什么？

**参考答案：** hidden 的 batch 维仍是旧 batch size，需要重新初始化或正确管理 hidden。

13. RNN 梯度爆炸怎么处理？

**参考答案：** 检查数据和学习率，规范化输入，降低学习率并使用 gradient clipping。

14. 训练准确但部署性能差，可能与什么有关？

**参考答案：** 训练/推理模式、teacher forcing 与真实自回归差异、padding、数据分布偏移或指标不匹配。

15. 为什么长序列训练显存大？

**参考答案：** 需要保存许多时间步的中间状态和计算图以反向传播。

16. truncated BPTT 解决什么问题？

**参考答案：** 把长序列分段反向传播，降低显存和计算成本，但会限制跨窗口的梯度传播。

17. 如何判断研究代码用 LSTM 的最后状态还是所有输出？

**参考答案：** 看后续 head 的输入 shape、任务是 many-to-one 还是 many-to-many，以及是否有 mask。

18. LSTM 作为 World Model dynamics 有什么限制？

**参考答案：** 它依赖递归状态，长上下文和并行计算能力有限，误差也可能逐步累积。

19. 为什么必须明确 time 与 batch？

**参考答案：** 维度交换可能不报错却改变语义，模型会把样本当时间步或反过来。

20. 调试时序模型的第一批打印项是什么？

**参考答案：** 输入 shape、length/mask、hidden shape、输出 shape、loss mask 和梯度范数。

## 15. 掌握度检查（附答案）

1. 序列、时间步、特征分别是什么？

**答案：** 序列是一组有序元素，时间步是其中一个位置，特征是该位置的数值描述。

2. [B,T,F] 如何解释？

**答案：** batch 数、时间长度、每个时间步的特征维度。

3. hidden state 为什么需要递归传递？

**答案：** 它把过去信息带到当前时间步。

4. LSTM 的 cell state 与 hidden state 有何区别？

**答案：** cell state 主要承担长期信息传递，hidden state 是当前对外暴露的状态表示。

5. many-to-many 适合什么场景？

**答案：** 每个时间步都需要输出的任务，如逐步动作或序列标注。

6. padding 为什么会造成问题？

**答案：** padding 不是实际数据，会污染最后状态和 loss。

7. mask 的作用是什么？

**答案：** 标记有效时间步，使计算只使用真实序列内容。

8. teacher forcing 与推理有何差别？

**答案：** 训练可使用真实上一时刻目标，推理只能使用模型自身历史预测。

9. 为什么会有梯度消失/爆炸？

**答案：** 长序列中多次链式求导会使梯度反复缩小或放大。

10. batch_first 的意义是什么？

**答案：** 指定输入输出维度把 batch 放在第一维。

11. LSTM 在具身智能中的价值是什么？

**答案：** 用历史观测和动作构造当前时刻的记忆表示，处理部分可观测性。

12. 何时查 [[Attention]]？

**答案：** 需要理解非递归地选择序列中重要位置、Q/K/V 或长上下文时。

13. 何时查 [[损失函数]]？

**答案：** 需要设计序列 mask、每步 loss、动作预测或多任务目标时。

14. 如何判断是否应该截断反向传播？

**答案：** 结合序列长度、显存、任务需要的长期依赖和训练稳定性判断。

15. 读 LSTM research code 最先看什么？

**答案：** batch/time 顺序、hidden 初始化、padding 处理、输出取法和训练推理差异。
