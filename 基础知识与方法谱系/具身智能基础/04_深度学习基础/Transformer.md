# Transformer：用注意力构建现代 AI 模型

Transformer 是以注意力为核心的深度学习架构。它不依赖传统 RNN 的逐步递归，而是通过 self-attention 在序列位置之间建立关系，再用前馈网络、残差连接和归一化堆叠成 block。

Transformer 在语言模型、视觉 Transformer、视频模型、机器人策略、轨迹建模、World Model 和多模态模型中都很常见。本文解释一个 Transformer block 如何工作；Attention 的 Q/K/V 细节见 [[Attention]]，训练机制见 [[损失函数]]、[[优化器]] 和 [[PyTorch]]。

## 0. 学习目标

学完以后能够：

- 解释 Transformer 为什么适合长序列和多模态 token；
- 读懂 embedding、position、attention、FFN、residual、LayerNorm；
- 追踪 Transformer 中常见的 shape；
- 区分 encoder、decoder、causal self-attention 和 cross-attention；
- 看懂视觉、语言和机器人 research code 的模块边界；
- 排查 mask、位置编码、维度、显存和训练推理不一致。

## 1. 为什么需要 Transformer？

RNN 一步一步处理序列，长距离依赖需要经过许多递归步骤；CNN 需要堆叠局部窗口才能扩大关系范围。Transformer 让每个位置直接与其他位置建立注意力关系，并可以并行处理训练序列。

```text
输入 token / patch / state
↓
Embedding + Position
↓
Transformer Block × L
  ├─ Self-Attention
  ├─ Residual + LayerNorm
  ├─ Feed-Forward Network
  └─ Residual + LayerNorm
↓
Task Head
↓
loss / prediction
```

并行不代表没有顺序概念：Transformer 必须通过位置编码或位置机制知道顺序。

## 2. 核心概念

### 2.1 Token 与 Embedding

Token 是序列中的一个基本单位。文本中可以是词片段，图像中可以是 patch，机器人中可以是某个时间步的状态、动作或多模态组合。

Embedding 是把离散 token 或原始特征映射到统一的模型维度 d_model 的向量。

```text
token ids / image patches / robot states
↓
embedding
↓
[B,T,d_model]
```

### 2.2 Position

Self-attention 本身对输入位置没有天然顺序概念。位置编码（Positional Encoding）或位置嵌入把位置信息加入 token，使模型区分第一个和第二个位置。

具体方法包括绝对位置、相对位置和 rotary position embedding。当前先理解它们都在解决“模型如何知道顺序”这一问题。

### 2.3 Transformer Block

一个常见 pre-norm block：

```text
x
↓
LayerNorm
↓
Self-Attention
↓
x + attention_output
↓
LayerNorm
↓
FFN
↓
x + ffn_output
```

残差连接让输入可以直接流到后面，帮助深层网络训练；LayerNorm 稳定每个 token 的特征尺度；FFN 对每个位置单独做非线性变换。

### 2.4 Feed-Forward Network

FFN 常见形式：

$$
FFN(x)=W_2 GELU(W_1x+b_1)+b_2
$$

它通常先把 d_model 扩大到 d_ff，再投影回来。Attention 混合位置之间的信息，FFN 主要在每个位置内部做特征变换。

### 2.5 Encoder、Decoder 与 Decoder-only

- encoder：双向读取输入，适合表示和理解；
- decoder：生成输出，常带 causal mask；
- encoder-decoder：encoder 编码条件，decoder 逐步生成；
- decoder-only：用 causal self-attention 进行自回归建模。

具体项目可能改造这些结构，不能只根据类名猜角色，要看 mask 和数据流。

## 3. Shape 追踪

设 B=32，序列长度 T=128，模型维度 D=256：

```text
输入 embedding: [32,128,256]
Q/K/V: [32,heads,128,head_dim]
attention output: [32,128,256]
FFN 中间: [32,128,1024]
block 输出: [32,128,256]
```

残差相加的两个张量 shape 必须相同。多头 attention 的 heads × head_dim 通常等于 D。

## 4. 一个最小 Transformer 编码器

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    batch_first=True,
    norm_first=True,
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

x = torch.randn(32, 128, 256)
y = encoder(x)
print(y.shape)  # [32, 128, 256]
```

参数解释：

- d_model：每个 token 的特征维度；
- nhead：attention head 数；
- dim_feedforward：FFN 中间宽度；
- num_layers：block 重复次数；
- batch_first：输入使用 [B,T,D]；
- norm_first：采用 pre-norm 变体。

## 5. Decoder 和 causal prediction

语言模型或动作序列生成通常不能看到未来目标。causal mask 形成下三角可见关系：

```text
第 t 个位置只能使用 1...t 的信息
不能使用 t+1...T
```

训练时可以一次并行计算所有位置，只要 mask 正确；推理时通常逐步生成，并可能缓存过去的 Key/Value。

在机器人轨迹建模中，必须确认动作和观测的时间对齐：预测 a_t 时能看到哪些 o_≤t，取决于任务定义。

## 6. 视觉与具身智能中的 Transformer

### 6.1 Vision Transformer

把图像切成 patch，展平并投影为 token：

```text
image [B,3,H,W]
↓
patchify
↓
patch tokens [B,T,D]
↓
Transformer
↓
class token 或 pooled feature
```

### 6.2 机器人轨迹

将观测、动作、语言目标或时间编码为 token，Transformer 建模它们之间的关系，再由 head 输出动作、价值或下一状态。

### 6.3 World Model

Transformer 可以根据历史 latent、action 和 observation 预测未来 latent，或为 planning 提供序列预测器。重要的是明确：

- token 表示什么；
- context 长度是多少；
- causal 方向是什么；
- 预测目标错开了几个时间步；
- loss 在哪些位置计算。

## 7. 训练、评估与 checkpoint

Transformer 通常参数量和显存较大。训练时关注：

- batch size、序列长度、head 数、层数；
- mixed precision 和梯度累积；
- learning rate warmup 与 scheduler；
- gradient clipping；
- train/eval；
- 保存模型、optimizer、scheduler 和随机状态。

这些属于 [[优化器]]、[[PyTorch]] 和 [[模型评估]] 的职责，本篇只建立整体联系。

## 8. 常见错误与 Debug

### 8.1 d_model 与 head 不整除

若 d_model=256、nhead=8，每个 head 为 32；若不能整除，通常直接报错或结构定义不合理。

### 8.2 忘记位置机制

模型可能能拟合局部模式，却无法正确利用顺序。检查位置嵌入是否创建、长度是否覆盖、推理长度是否超出训练范围。

### 8.3 causal mask 方向反了

模型若能看到未来，训练 loss 可能异常漂亮，部署时却失效。用很短的序列手工检查可见矩阵。

### 8.4 padding 没有屏蔽

变长序列 padding 会污染 attention 和 loss；需要 key padding mask 与 loss mask 保持一致。

### 8.5 序列太长导致显存爆炸

attention 的 token 配对带来平方级成本。先确认真实 T，再减少 token、分块、局部注意力或使用合适的缓存策略。

### 8.6 训练和推理时间对齐不同

训练用 shifted target，推理用上一时刻预测；若两者输入定义不同，会出现 exposure bias 或 off-by-one。

## 9. 容易混淆的概念

| 概念 | 含义 | 不等于 |
|---|---|---|
| token | 序列单位 | 必然是一个词 |
| d_model | token 表示维度 | 序列长度 |
| nhead | 注意力头数 | 层数 |
| block | 一组 attention/FFN 子层 | 一个时间步 |
| residual | 输入与子层输出相加 | 直接跳过训练 |
| causal mask | 禁止看未来 | padding mask |
| encoder | 编码输入 | 一定不能生成 |
| decoder | 生成或条件解码模块 | 只能用于文本 |

## 10. 与其他笔记的关系

前置：[[MLP]]、[[Attention]]、[[PyTorch]]、[[损失函数]]、[[优化器]]、[[模型评估]]。

相关：[[CNN]]、[[RNN_LSTM]]、[[表征学习]]、[[自监督学习]]。

本篇不展开 Transformer 的各种高效注意力、具体预训练目标和大型模型工程优化；当前知道它们存在即可。

## 11. 知识地图

```text
token / patch / state
↓
embedding + position
↓
attention：跨位置混合信息
↓
residual + normalization
↓
FFN：位置内非线性变换
↓
重复 block
↓
task head
↓
loss、评估、部署
```

## 12. 快速查询表

| 看到的变量 | 先问什么 |
|---|---|
| d_model | 每个 token 多少维 |
| nhead | 能否整除 d_model |
| max_seq_len | 实际 T 是否超出 |
| mask | 是否允许看到未来/ padding |
| qkv | 来源是否 self 或 cross |
| logits | 接什么 loss、时间是否错位 |
| checkpoint | 是否保存 optimizer/scheduler |

## 13. 练习题（题目下附参考答案）

1. Transformer 的核心机制是什么？

**参考答案：** self-attention，通过 token 之间的相关性动态混合信息。

2. 为什么需要位置编码？

**参考答案：** 注意力本身没有天然顺序概念，位置机制让模型区分不同位置。

3. d_model 表示什么？

**参考答案：** 每个 token 的模型表示维度。

4. residual connection 做什么？

**参考答案：** 将子层输入与输出相加，提供直接信息和梯度通路，帮助深层训练。

5. FFN 主要做什么？

**参考答案：** 对每个位置的表示进行非线性特征变换，通常先扩维再投影回 d_model。

6. d_model=256、nhead=8 时 head_dim 是多少？

**参考答案：** 32，因为 256/8=32。

7. 输入 [32,128,256] 经过 encoder 后通常是什么 shape？

**参考答案：** [32,128,256]，序列长度和模型维度通常保持不变。

8. encoder 与 decoder 的主要差别是什么？

**参考答案：** encoder 通常可双向看输入，decoder 常使用 causal mask 进行自回归生成。

9. 训练时为什么可以并行计算 causal decoder？

**参考答案：** 因为 mask 阻止每个位置访问未来，所有位置可在一次前向中并行计算。

10. Transformer 在图像中如何得到 token？

**参考答案：** 将图像切成 patch，展平并线性投影为 patch token。

11. 机器人轨迹中 token 可以表示什么？

**参考答案：** 状态、动作、视觉 patch、语言目标或时间条件。

12. d_model 不能被 nhead 整除怎么办？

**参考答案：** 修改二者配置，使每个 head 有整数维度；不能靠 reshape 隐藏结构错误。

13. Transformer loss 异常低、部署失败，可能为什么？

**参考答案：** 可能 causal mask 反了、标签泄漏或训练推理时间对齐不一致。

14. 序列长度增加为什么显存增长快？

**参考答案：** attention 需要构造近似 T×T 的位置关系矩阵。

15. 变长轨迹如何处理？

**参考答案：** padding 后使用 key padding mask 和 loss mask，或采用其他变长处理方案。

16. 为什么预训练 Transformer 迁移到机器人数据时要检查归一化和 token 化？

**参考答案：** 输入分布和 token 语义改变会使预训练表示失配。

17. World Model 用 Transformer 预测未来时必须明确什么？

**参考答案：** token 定义、上下文长度、causal 方向、预测偏移和 loss mask。

18. 如何判断 attention 是 self 还是 cross？

**参考答案：** 查看 Q 与 K/V 的来源是否相同。

19. Transformer 训练显存不足时先查什么？

**参考答案：** 序列长度、batch、层数、隐藏维度和 attention 实际 shape，再考虑降 token、分块和混合精度。

20. 为什么只看 token-level loss 不足以判断机器人成功？

**参考答案：** 低层预测准确不一定转化为闭环控制成功，还要看任务级指标和未见场景。

## 14. 掌握度检查（附答案）

1. Transformer 为什么能并行处理训练序列？

**答案：** self-attention 可同时计算各位置，且 causal mask 在需要时限制未来信息。

2. 位置编码解决什么问题？

**答案：** 向没有天然顺序的注意力输入注入位置信息。

3. attention 与 FFN 的职责分别是什么？

**答案：** attention 混合不同位置的信息，FFN 在每个位置内部做非线性变换。

4. residual 为什么重要？

**答案：** 提供直接的信息和梯度路径，缓解深层网络优化困难。

5. d_model、nhead、head_dim 的关系是什么？

**答案：** 通常 d_model=nhead×head_dim。

6. causal mask 防止什么？

**答案：** 防止当前位置访问未来 token，避免训练目标泄漏。

7. encoder、decoder-only 的典型用途是什么？

**答案：** encoder 常用于输入表示，decoder-only 常用于自回归序列生成；实际项目仍需看具体实现。

8. Transformer 如何表示一张图？

**答案：** 将图像切为 patch，再把 patch 映射成 token 序列。

9. 机器人中的 Transformer token 可以是什么？

**答案：** 观测、动作、图像 patch、语言指令、目标和时间信息。

10. 为什么序列长度是工程瓶颈？

**答案：** attention 的计算和显存通常随长度平方增长。

11. 如何排查 mask？

**答案：** 打印 mask 和 score shape，手工验证短序列的可见位置，确认布尔语义。

12. 如何排查时间错位？

**答案：** 明确输入时间戳、目标时间戳、shift 操作和推理时使用的历史，打印一个样本对齐检查。

13. Transformer 与 CNN 的主要差异是什么？

**答案：** CNN 先验强调局部空间和共享卷积，Transformer 用注意力动态建立全局或跨模态关系。

14. 什么时候查 [[Attention]]？

**答案：** 需要追踪 Q/K/V、Softmax、mask 和注意力权重 shape 时。

15. 读 Transformer research code 最先看哪五件事？

**答案：** token 来源、position 机制、block 结构、mask/时间方向、输出 head 与 loss。
