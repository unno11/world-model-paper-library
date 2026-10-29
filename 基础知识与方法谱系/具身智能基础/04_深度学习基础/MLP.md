# MLP：最基础的神经网络结构

多层感知机（Multi-Layer Perceptron，MLP）是理解深度学习模型的起点。它把输入向量经过若干个线性层和非线性激活函数，逐步变换成预测结果。MLP 常见于分类头、策略网络、价值网络、状态预测器和 Transformer 的前馈子层。

本文重点负责层、参数、激活函数、前向传播和 shape；损失与更新分别链接到 [[损失函数]]、[[优化器]] 和 [[PyTorch]]。

## 0. 学习目标

学完以后能够：

- 解释神经网络、层、参数、权重、偏置和激活函数；
- 从输入输出 shape 推断 MLP 结构；
- 读懂 Linear → Activation → Linear；
- 区分前向计算、反向传播和参数更新；
- 调试维度不匹配、输出范围错误和过拟合；
- 看懂 MLP 在 AI research repository 中的角色。

## 1. 为什么需要 MLP？

线性模型只能计算 Wx+b。很多图像特征、机器人状态和动作之间不是简单直线关系。MLP 通过多层变换和非线性激活学习更复杂的函数。

```text
输入 x
↓
线性变换 Wx+b
↓
非线性激活
↓
再次线性变换
↓
预测结果
```

只堆叠线性层仍然等价于一个线性层，因此隐藏层通常需要 ReLU 或 GELU。

## 2. 核心概念

### 2.1 神经网络与层

神经网络是由参数化计算模块组成的模型。层（Layer）是其中一个模块，例如一个线性层把最后一维从 4 变成 8。

```python
import torch
from torch import nn

layer = nn.Linear(4, 8)
x = torch.randn(32, 4)
y = layer(x)
print(y.shape)  # [32, 8]
```

32 是 batch size，4 是每个样本的输入特征数，8 是输出特征数。

### 2.2 参数、权重和偏置

参数（Parameter）是训练时会被调整的数值。线性层可以写成 y=xWᵀ+b：

- weight 决定输入特征如何组合；
- bias 提供平移；
- 梯度告诉我们参数应向哪个方向改变；
- optimizer 根据梯度执行更新。

### 2.3 激活函数

激活函数（Activation Function）是加入非线性的函数。ReLU 计算 max(0,x)，GELU 在 Transformer 中常见。代码和论文中通常直接使用 ReLU、GELU 这些英文名。

### 2.4 隐藏层与表示

输入和输出之间的中间层叫隐藏层。隐藏层输出是模型学习的中间表示，不是人工标签。宽度是单层单元数，深度是层数，容量是拟合复杂规律的能力。

## 3. 整体工作原理

两层 MLP：

$$
h = ReLU(W_1x+b_1)
$$

$$
\hat{y}=W_2h+b_2
$$

```text
输入特征
↓
Linear(in, hidden)
↓
ReLU 或 GELU
↓
Linear(hidden, out)
↓
预测
↓
loss
↓
backward 与 optimizer.step
```

MLP 本身通常只负责预测；loss 和参数更新属于训练流程。

## 4. 从零读代码

```python
class ActionMLP(nn.Module):
    def __init__(self, obs_dim=12, hidden_dim=128, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

model = ActionMLP()
obs = torch.randn(32, 12)
action = model(obs)
print(action.shape)  # [32, 7]
```

obs_dim 是每个观测的维度，hidden_dim 是中间表示维度，action_dim 是动作维度。输入最后一维 12，输出最后一维 7，batch 维 32 保留。

## 5. 训练中各步骤的职责

```python
optimizer.zero_grad()
pred = model(obs)
loss = loss_fn(pred, target)
loss.backward()
optimizer.step()
```

- zero_grad：清理上一轮累积的梯度；
- model：前向计算；
- loss_fn：得到可优化的标量；
- backward：计算梯度；
- step：修改参数。

详细训练循环见 [[PyTorch]]，选择 loss 见 [[损失函数]]，学习率和更新规则见 [[优化器]]。

## 6. 在真实 AI 项目中的位置

| 位置 | 输入 | 输出 |
|---|---|---|
| 分类头 | CNN 或 Transformer 特征 | 类别 logits |
| 策略网络 | 机器人状态或视觉特征 | 动作 |
| 价值网络 | 状态或状态-动作 | 标量价值 |
| 预测头 | latent state 与 action | 下一状态 latent |
| 融合模块 | 多模态特征 | 统一表示 |

典型流程是：读取 config → 构造 DataLoader → 编码输入 → MLP 预测 → 计算 loss → 更新 → 验证 → 保存 checkpoint。

在 World Model 中，MLP 可能预测下一 latent、奖励或终止标志；在行为克隆中，MLP 可能把观测直接映射为动作。

## 7. 常见错误与 Debug

### 7.1 维度不匹配

现象：矩阵维度无法相乘。

原因：输入最后一维不等于第一层的 in_features。

判断：

```python
print(obs.shape)
print(model.net[0].in_features)
```

解决：追踪数据预处理、flatten、transpose、squeeze 和 Linear 定义；不要盲目 reshape。

### 7.2 输出头选错

分类通常输出 logits，交给 CrossEntropyLoss；连续回归通常输出实数，使用 MSE 或 Huber。最后是否使用 sigmoid 或 tanh，要与目标范围和 loss 一致。

### 7.3 训练好、验证差

可能是过拟合、数据泄漏或训练验证预处理不一致。结合 [[正则化]] 和 [[模型评估]] 看曲线及未见数据。

### 7.4 输出全为 NaN

检查输入、loss、学习率、梯度范数和数据中的 NaN/Inf；必要时降低学习率、规范化输入、裁剪梯度。

## 8. 容易混淆

| 概念 | 含义 | 不等于 |
|---|---|---|
| 层 | 一次参数化计算 | 一个训练 step |
| 参数 | 会学习的数值 | 输入数据 |
| 激活 | 非线性变换 | loss |
| logits | 未归一化分数 | 概率 |
| forward | 产生预测 | 更新参数 |
| width | 一层单元数 | batch size |

## 9. 与其他笔记的关系

前置：[[监督学习]]、[[损失函数]]、[[优化器]]、[[正则化]]、[[模型评估]]、[[PyTorch]]。

相关结构：[[CNN]]、[[RNN_LSTM]]、[[Attention]]、[[Transformer]]。

本篇不展开卷积空间结构、序列记忆和注意力矩阵，分别见对应笔记；表示学习见 [[表征学习]] 和 [[自监督学习]]。

## 10. 知识地图

```text
输入
↓
Linear 改变维度
↓
Activation 引入非线性
↓
隐藏表示
↓
输出头
↓
loss
↓
梯度与优化器
↓
验证泛化
```

## 11. 快速查询表

| 任务 | 方法 | 重点 |
|---|---|---|
| 改变维度 | Linear(a,b) | 输入最后一维为 a |
| 加非线性 | ReLU 或 GELU | 通常放隐藏层 |
| 看输出 | 打印 shape | batch 是否保留 |
| 训练 | forward → loss → backward → step | 参数是否变化 |
| 查过拟合 | 比较 train/val 曲线 | 不只看训练 loss |

## 12. 练习题（题目下附参考答案）

1. MLP 的两类核心操作是什么？

**参考答案：** 线性变换和非线性激活。前者组合特征，后者让整体函数具有非线性表达能力。

2. Linear(10,4) 的输入和输出最后一维是什么？

**参考答案：** 输入为 10，输出为 4。

3. 参数是什么？

**参考答案：** 训练时被梯度和优化器调整的数值，例如 weight 和 bias。

4. 只堆叠多个线性层能得到非线性模型吗？

**参考答案：** 不能。多个线性变换仍可合并为一个线性变换。

5. forward 会更新参数吗？

**参考答案：** 不会。forward 只计算预测，通常由 optimizer.step 更新。

6. 隐藏层越宽一定越好吗？

**参考答案：** 不一定。宽度增加容量，也可能造成过拟合、显存增加和训练变慢。

7. 输入 shape 为 [64,20]，经过 Linear(20,50) 后是什么？

**参考答案：** [64,50]，Linear 作用于最后一维。

8. Linear(8,3) 的输出是否直接是三类概率？

**参考答案：** 通常不是，而是 logits；多分类一般交给 CrossEntropyLoss。

9. 以下哪一步真正修改参数：forward、backward、step？

**参考答案：** step。backward 计算梯度，step 使用梯度更新。

10. 12 维状态预测 7 维连续动作，MLP 首尾维度如何设？

**参考答案：** 第一层输入为 12，最后一层输出为 7，例如 Linear(12,128) → ReLU → Linear(128,7)。

11. 分类头和动作回归头为何可能不同？

**参考答案：** 分类输出 logits 并用分类损失，动作回归输出实数并用 MSE/Huber；输出范围处理也可能不同。

12. CNN 输出 [32,512]，10 类分类头怎么写？

**参考答案：** 最后一层 Linear(512,10)，输出 [32,10] 的 logits。

13. 维度报错时第一步检查什么？

**参考答案：** 输入张量 shape、Linear 的 in_features，以及中间是否错误 flatten 或交换了维度。

14. 训练 loss 降低、验证 loss 升高说明什么？

**参考答案：** 可能过拟合，也要检查数据泄漏和训练/验证预处理差异。

15. MLP 输出 NaN 时如何排查？

**参考答案：** 依次检查输入、loss、学习率、梯度和数据中的 NaN/Inf，再采取降低学习率、规范化或裁剪梯度。

16. policy 输出多了一个维度，怎么定位？

**参考答案：** 打印观测、模型输出和目标动作 shape，检查 DataLoader、unsqueeze、squeeze 和最后一层定义。

17. 如何判断 Sequential 是分类头还是状态预测器？

**参考答案：** 看输入来源、输出维度、后接的 loss 和调用位置；不能只根据变量名判断。

18. 冻结 backbone 只训练 MLP 时检查什么？

**参考答案：** backbone 的 requires_grad、optimizer 参数集合、模型模式和特征是否意外 detach。

19. MLP 在 World Model 中可以预测什么？

**参考答案：** 下一 latent、奖励、终止标志或其他辅助目标。

20. 为什么不能只看训练 loss 判断机器人策略有效？

**参考答案：** 训练 loss 不能代表未见数据和闭环执行效果，还需验证、测试、仿真或真实执行评估。

## 13. 掌握度检查（附答案）

1. 为什么需要激活函数？

**答案：** 让层叠网络具有非线性表达能力；否则多层 Linear 仍等价于单个线性变换。

2. Linear(a,b) 中 a、b 分别是什么？

**答案：** 输入和输出的最后一维大小。

3. 参数、梯度、优化器的关系是什么？

**答案：** 参数是要学习的数值，梯度描述 loss 对参数的变化，优化器利用梯度更新参数。

4. batch 和 feature 的区别是什么？

**答案：** batch 是样本数，feature 是每个样本的特征数。

5. logits 与概率有什么区别？

**答案：** logits 是未归一化分数，概率是经过相应归一化后的数值。

6. 一轮训练的主要顺序是什么？

**答案：** zero_grad → forward → loss → backward → step。

7. 输出维度由什么决定？

**答案：** 由任务目标决定，如类别数、动作维度或预测状态维度。

8. 如何判断 shape mismatch？

**答案：** 比较输入最后一维与 Linear 的 in_features，并沿数据流追踪 shape。

9. 为什么大 MLP 可能过拟合？

**答案：** 参数和函数容量较大，可能记忆训练样本而不能泛化。

10. 什么时候查损失函数笔记？

**答案：** 需要选择回归/分类 loss、理解 logits、reduction 或多任务 loss 时。

11. 什么时候查优化器笔记？

**答案：** 需要理解学习率、梯度、权重衰减、参数更新或训练不稳定时。

12. MLP 在具身智能中的常见用途是什么？

**答案：** 策略、价值、状态预测、latent 预测和特征融合。

13. 如何判断过拟合？

**答案：** 看训练与验证曲线以及未见任务表现；训练变好而验证变差是典型信号。

14. 冻结 backbone 时 optimizer 应包含什么？

**答案：** 通常只包含需要训练的 MLP head 参数。

15. 阅读研究代码中的 MLP，最先看什么？

**答案：** 输入来源、输入输出 shape、后接的 loss 或模块。
