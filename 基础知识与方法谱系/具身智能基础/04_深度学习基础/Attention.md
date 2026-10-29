# Attention：让模型选择序列中重要的信息

注意力（Attention）是一种根据当前需求，对输入不同位置分配不同权重的机制。它解决了一个核心问题：当输入很长时，模型不应把所有位置等量处理，而应动态寻找与当前计算最相关的信息。

本文重点负责 Query、Key、Value、相似度、权重、mask 和 self-attention；Transformer 的完整 block、位置编码和训练方式见 [[Transformer]]。损失、优化与张量实现链接到 [[损失函数]]、[[优化器]]、[[PyTorch]]。

## 0. 学习目标

学完以后能够：

- 用直觉解释注意力为什么存在；
- 区分 Query、Key、Value；
- 读懂 scaled dot-product attention 的 shape；
- 理解 self-attention 与 cross-attention；
- 解释 causal mask 和 padding mask；
- 在视觉、语言、机器人轨迹和 World Model 代码中定位 attention。

## 1. 为什么需要注意力？

传统序列模型把历史压缩进一个固定状态，长序列可能丢失细节。注意力允许当前位置直接查看输入序列的多个位置，并按相关程度加权。

```text
当前查询 Query
↓
与所有 Key 比较相似度
↓
得到 attention weights
↓
对所有 Value 加权求和
↓
当前输出
```

注意力不是“理解”本身，而是一种可学习的信息路由机制。

## 2. 核心概念

### 2.1 Query、Key、Value

**Query（查询）**：当前计算想找什么。

**Key（键）**：每个候选位置用什么特征表示自己。

**Value（值）**：真正被取回并汇聚的内容。

可以把它类比为图书馆检索：Query 是问题，Key 是目录索引，Value 是找到的正文。

### 2.2 Scaled Dot-Product Attention

公式：

$$
Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V
$$

步骤：

1. Q 与 K 做点积，得到相关性分数；
2. 除以 sqrt(d_k)，避免维度大时分数过大；
3. 对最后一维做 Softmax，得到权重；
4. 用权重加权 V。

### 2.3 Shape

设 batch 为 B，查询长度为 Tq，键值长度为 Tk，特征维为 D：

- Q: [B,Tq,D]
- K: [B,Tk,D]
- V: [B,Tk,D]
- QKᵀ: [B,Tq,Tk]
- 输出: [B,Tq,Dv]

其中 Tq 和 Tk 可以不同，这正是 cross-attention 有用的原因。

### 2.4 Self-Attention

self-attention 中 Q、K、V 来自同一序列。序列中的每个位置都可以根据其他位置更新自己的表示。

例如机器人轨迹中，一个时间步的动作表示可以参考附近观测、目标信息和早先动作。

### 2.5 Cross-Attention

cross-attention 中 Query 与 Key/Value 来自不同序列。例如：

- 图像 Query 关注文本特征；
- 机器人状态 Query 关注视觉 patch；
- decoder Query 关注 encoder 输出。

### 2.6 Multi-Head Attention

多头注意力（Multi-Head Attention）把特征分成多个 head，每个 head 在不同子空间计算注意力，再拼接和投影。

不同 head 可能学习时间关系、空间关系或语义关系，但不能简单地把每个 head 人工解释成固定含义。

## 3. Attention 的整体流程

```text
输入序列
↓
线性投影得到 Q、K、V
↓
Q 与 K 计算分数
↓
加 mask
↓
Softmax 得权重
↓
加权汇聚 V
↓
输出投影
↓
残差连接与后续模块
```

在 Transformer 中，attention 通常与残差连接、LayerNorm 和前馈网络共同构成 block，见 [[Transformer]]。

## 4. 从零读一个最小实现

```python
def attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v
```

逐行解释：

- q 与 k 转置后相乘，最后两维形成相关性矩阵；
- mask 为 0 的位置被设为负无穷，Softmax 后权重接近 0；
- softmax(dim=-1) 在候选 Key 位置上归一化；
- weights @ v 对 Value 加权。

测试 shape：

```python
q = torch.randn(2, 4, 8)
k = torch.randn(2, 6, 8)
v = torch.randn(2, 6, 16)
out = attention(q, k, v)
print(out.shape)  # [2, 4, 16]
```

2 是 batch，4 是 Query 长度，6 是 Key/Value 长度，输出 Value 维度为 16。

## 5. Mask：告诉模型哪些位置不能看

### 5.1 Padding Mask

变长序列补齐后，padding 位置不是真实内容。padding mask 防止模型把它当信息读取。

### 5.2 Causal Mask

自回归生成中，位置 t 不能看到未来位置 t+1。causal mask 保持时间方向：

```text
位置 1：只能看 1
位置 2：可以看 1、2
位置 3：可以看 1、2、3
```

### 5.3 Mask 的方向

不同库可能使用 True 表示屏蔽，也可能 True 表示允许。研究代码中要阅读 API 文档和调用处，不能凭记忆猜。

## 6. 复杂度与局部注意力

标准 self-attention 的分数矩阵大小约为 T×T，时间和显存随序列长度平方增长。长视频、长轨迹和高分辨率视觉 token 会带来压力。

常见思路：

- window/local attention；
- downsample token；
- sparse attention；
- chunking；
- memory 或缓存；
- 先用 CNN 降低空间 token 数。

目前只需知道这些存在；具体高效实现属于后续专题。

## 7. 在 AI 与具身智能中的使用

Attention 常出现在：

- Transformer 文本和视觉编码器；
- 多视角图像或视频 token 融合；
- 机器人观测—动作轨迹建模；
- 语言条件策略；
- World Model 的历史 latent 聚合；
- planning 中从候选状态选择相关信息。

典型流程：

```text
image patches / state tokens / action tokens
↓
self-attention 建立同类 token 关系
↓
cross-attention 融合语言、目标或视觉条件
↓
Transformer block
↓
policy / dynamics / value head
```

## 8. 常见错误与 Debug

### 8.1 Softmax 维度错误

权重应在候选 Key 维归一化，通常是最后一维。若在 query 维 Softmax，含义会改变。

### 8.2 Q/K 特征维不匹配

Q 和 K 做点积时最后一维必须相同，或者先通过投影映射到相同 head_dim。

### 8.3 mask shape 广播错误

先打印 scores 和 mask shape，确认 mask 能广播到 [B,heads,Tq,Tk]，并确认 True/False 语义。

### 8.4 忘记缩放

不除以 sqrt(d_k) 可能使 logits 过大，Softmax 过早饱和，梯度变差。

### 8.5 看见 attention map 就过度解释

注意力权重不一定等于因果解释或人类可读的重要性。还要结合遮挡实验、下游指标和多个样本。

## 9. 容易混淆的概念

| 概念 | 含义 | 不等于 |
|---|---|---|
| Query | 当前需求 | 标签 |
| Key | 候选匹配特征 | 被输出的内容 |
| Value | 被加权汇聚的内容 | attention weight |
| score | 匹配分数 | 概率 |
| weight | Softmax 后权重 | 模型参数 weight |
| self-attention | QKV 来自同一序列 | 只能用于文本 |
| causal mask | 屏蔽未来 | padding mask |

## 10. 与其他笔记的关系

前置：[[PyTorch]]、[[MLP]]、[[损失函数]]、[[优化器]]、[[模型评估]]。

后续：[[Transformer]]、[[表征学习]]、[[自监督学习]]。

本篇不完整讲 Transformer 的位置编码、LayerNorm、残差和 FFN；这些由 [[Transformer]] 负责。

## 11. 知识地图

```text
Q、K、V
↓
QKᵀ 相似度
↓
缩放
↓
mask
↓
Softmax
↓
加权 V
↓
多头拼接与投影
↓
Transformer block
```

## 12. 快速查询表

| 检查项 | 正常含义 |
|---|---|
| Q shape | [B,Tq,D] |
| K shape | [B,Tk,D] |
| V shape | [B,Tk,Dv] |
| score shape | [B,Tq,Tk] |
| Softmax 维度 | 候选 Key 的最后一维 |
| causal mask | 当前不能看到未来 |
| padding mask | 忽略补齐位置 |

## 13. 练习题（题目下附参考答案）

1. 注意力解决的核心问题是什么？

**参考答案：** 让模型根据当前需求动态选择输入序列中更相关的位置，而不是把所有信息固定压缩或等量处理。

2. Query、Key、Value 分别是什么？

**参考答案：** Query 表示想找什么，Key 表示候选如何被匹配，Value 是实际被取回并汇聚的内容。

3. 注意力公式中 Softmax 的作用是什么？

**参考答案：** 把匹配分数转成归一化权重，便于对 Value 加权求和。

4. self-attention 的 QKV 来自哪里？

**参考答案：** 来自同一个输入序列。

5. cross-attention 有什么不同？

**参考答案：** Query 与 Key/Value 来自不同序列，可用于跨模态或条件信息融合。

6. Q=[B,4,8]、K=[B,6,8] 时 score shape 是什么？

**参考答案：** [B,4,6]，表示每个 Query 位置对 6 个 Key 位置的分数。

7. V=[B,6,16] 时输出 shape 是什么？

**参考答案：** [B,4,16]。

8. 为什么要除以 sqrt(d_k)？

**参考答案：** 防止点积分数随维度增大而过大，使 Softmax 过度饱和、梯度变差。

9. causal mask 做什么？

**参考答案：** 屏蔽未来位置，保证自回归预测不能偷看未来目标。

10. padding mask 做什么？

**参考答案：** 忽略为了对齐长度而补出的无效位置。

11. 为什么 Softmax 维度通常是最后一维？

**参考答案：** 最后一维通常对应候选 Key 位置，注意力权重应在这些候选上归一化。

12. attention map 能直接证明模型因果关系吗？

**参考答案：** 不能。权重是信息分配的一部分，不等于充分的因果解释。

13. Q/K 最后一维不匹配怎么办？

**参考答案：** 检查投影层和 head_dim，使 Q、K 最后一维一致。

14. mask 报 shape error 怎么调试？

**参考答案：** 打印 scores、mask 及 batch/head/query/key 维，检查广播和 True/False 语义。

15. 长序列为什么让 attention 显存变大？

**参考答案：** score 矩阵随序列长度形成 T×T，长度增加会带来平方级增长。

16. 机器人策略中 attention 可以关注什么？

**参考答案：** 历史观测、目标、视觉 patch、动作 token 或其他传感器信息。

17. 多头注意力的好处是什么？

**参考答案：** 不同 head 可以在不同表示子空间学习不同关系，再组合成输出。

18. 为什么 attention 之前可能先用 CNN？

**参考答案：** CNN 可以先提取局部空间特征、降低视觉 token 数，再让 attention 建模全局关系。

19. 如何判断代码是 self-attention 还是 cross-attention？

**参考答案：** 查看 Q 的来源是否与 K/V 相同；相同是 self，不同通常是 cross。

20. World Model 中 attention 可能聚合什么？

**参考答案：** 历史 latent、动作、观测、任务条件或候选未来状态的信息。

## 14. 掌握度检查（附答案）

1. 注意力的直觉是什么？

**答案：** 根据当前需求对不同输入位置分配不同关注权重。

2. Q/K/V 的职责是什么？

**答案：** Q 表示查询需求，K 用于匹配，V 提供被汇聚内容。

3. score 矩阵的两个长度维代表什么？

**答案：** Query 位置数和 Key 位置数。

4. self 与 cross attention 的区别是什么？

**答案：** self 的 QKV 来自同一序列，cross 的 Q 与 K/V 来自不同序列。

5. 为什么需要 mask？

**答案：** 屏蔽 padding、未来或任务不允许访问的位置。

6. causal mask 与 padding mask 区别是什么？

**答案：** causal 防止看未来，padding 忽略补齐内容。

7. 为什么使用缩放点积？

**答案：** 控制点积分数尺度，避免 Softmax 过度饱和。

8. 多头注意力是什么？

**答案：** 将特征分成多个子空间分别计算注意力，再拼接和投影。

9. attention 的复杂度为什么与 T² 有关？

**答案：** 每个 Query 需要和所有 Key 配对，形成 T×T 分数矩阵。

10. Softmax 应在哪个维度做？

**答案：** 通常沿候选 Key 位置维度，也就是 score 的最后一维。

11. attention 权重能否当作因果解释？

**答案：** 不能单独这样判断，需要结合干预或下游实验。

12. Attention 在机器人研究中有什么价值？

**答案：** 聚合历史、视觉、动作、语言和目标条件，选择与当前决策相关的信息。

13. 什么时候查 [[Transformer]]？

**答案：** 需要理解完整 block、位置编码、残差、LayerNorm 和 FFN 时。

14. 调试 attention 最先打印什么？

**答案：** Q/K/V、score、mask、weight 和输出的 shape，以及 Softmax 后权重和是否正常。

15. 读 attention research code 最先找什么？

**答案：** QKV 来源、head 数与 head_dim、mask 构造、Softmax 维度和输出接入位置。
