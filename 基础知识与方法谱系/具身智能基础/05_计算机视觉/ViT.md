# ViT：把图像变成 Token 序列

Vision Transformer（ViT）把图像切成 patch，将每个 patch 映射为 token，再使用 Transformer 建模 patch 之间的关系。它把“视觉问题”转换为“序列建模问题”。

本文重点负责 patch embedding、位置编码、视觉 token、分类 token、分辨率与迁移；通用 Transformer 结构见 [[Transformer]]，注意力细节见 [[Attention]]，图像数据管线见 [[图像基础]]。

## 0. 学习目标

学完以后能够：

- 解释 ViT 为什么把图像切成 patch；
- 计算 patch 数量和 token shape；
- 读懂 patch embedding、class token 和 position embedding；
- 理解 ViT 与 CNN 的视觉先验差异；
- 排查图像尺寸、位置编码、通道、mask 和显存错误；
- 看懂 ViT 在视觉表征、视频和机器人研究中的使用。

## 1. 为什么需要 ViT？

CNN 通过局部卷积建立空间关系，ViT 先把图像拆成若干 patch，再让 attention 动态建模全局关系。

```text
image [B,C,H,W]
↓
patchify
↓
patches [B,N,P²C]
↓
linear projection
↓
tokens [B,N,D]
↓
position embedding
↓
Transformer encoder
↓
class token / pooling
↓
task head
```

ViT 的代价是 token 数越多，attention 的计算越大；patch 越小，细节更丰富但计算更昂贵。

## 2. 核心概念

### 2.1 Patch

patch 是固定大小的图像小块。若图像 H=W=224，patch size P=16，则每边有 14 个 patch，总数 N=196。

若 H、W 不能被 P 整除，需要 padding、resize 或裁剪，具体取决于实现。

### 2.2 Patch Embedding

每个 patch 展平后维度为 P×P×C，再通过线性层映射到 D 维 token。

```text
patch [P,P,C]
↓ flatten
[P×P×C]
↓ linear
[D]
```

也常用 kernel_size=P、stride=P 的 Conv2d 一步完成 patch embedding。

### 2.3 Class Token 与 Pooling

有些 ViT 在 token 序列前添加可学习的 class token，最终用它做整图分类；其他实现可能对所有 patch token 做平均池化或使用特殊聚合头。

不要假设第一个 token 一定是 class token，要看代码。

### 2.4 Position Embedding

Attention 对 token 顺序没有天然空间概念，因此需要加入 2D 或展平后的位置编码。改变输入分辨率时，位置编码可能需要插值。

### 2.5 Token Shape

设输入 [B,3,224,224]，P=16，D=768：

- patch 数 N=14×14=196；
- 无 class token：[B,196,768]；
- 有 class token：[B,197,768]。

## 3. ViT 与 CNN 的差异

| 方面 | CNN | ViT |
|---|---|---|
| 局部先验 | 强，卷积窗口 | 较弱，依赖数据和位置 |
| 全局关系 | 需堆层扩大感受野 | attention 可直接建立 |
| 输入 token | feature map | patch sequence |
| 小数据 | 通常更有优势 | 可能需要预训练 |
| 长/高分辨率 | 卷积计算可控 | token 和 attention 成本高 |

实际模型常组合 CNN 与 Transformer，不能把它们看成完全互斥。

## 4. 从零读懂 patch embedding

```python
patch_embed = nn.Conv2d(
    in_channels=3,
    out_channels=768,
    kernel_size=16,
    stride=16,
)

x = torch.randn(8, 3, 224, 224)
y = patch_embed(x)
print(y.shape)  # [8, 768, 14, 14]

tokens = y.flatten(2).transpose(1, 2)
print(tokens.shape)  # [8, 196, 768]
```

Conv 的输出是空间网格 [B,D,14,14]，flatten(2) 把空间位置合并，transpose 后得到 [B,N,D]。

## 5. ViT 的整体流程

```text
RGB image
↓
patch embedding
↓
optional class token
↓
position embedding
↓
Transformer encoder blocks
↓
class token / mean pooling
↓
classification, dense prediction, policy or latent head
```

对 dense prediction、检测或分割，不能简单只保留一个 class token，因为需要空间 patch 表示。

## 6. 在 AI 与具身智能中的使用

ViT 可以作为：

- 视觉预训练 encoder；
- 多视角或多相机 patch 融合器；
- 视频模型的空间 token 编码器；
- 语言条件机器人策略的视觉模块；
- World Model 的图像 latent encoder；
- 图像 patch 与状态、动作 token 的融合器。

机器人场景要关注相机分辨率、实时性、空间定位和跨环境泛化。高准确率但延迟过大的 ViT 可能不适合闭环控制。

## 7. 迁移与微调

预训练 ViT 迁移时检查：

- 输入分辨率和 patch size；
- RGB/BGR、归一化；
- position embedding 是否需要插值；
- class head 是否替换；
- encoder 是否冻结；
- 机器人图像与预训练图像的分布差异。

冻结 ViT 只训练 head 可做 linear probe；微调时可使用更小学习率，见 [[视觉表征]]。

## 8. 常见错误与 Debug

### 8.1 patch 数量算错

输入尺寸、patch size、padding 和 class token 会共同决定 token 数。打印 patch embedding 输出和最终序列长度。

### 8.2 忘记位置编码

模型可能无法稳定区分空间位置。检查 position embedding shape 是否覆盖所有 token。

### 8.3 分辨率改变后位置编码不匹配

如果预训练是 224、下游是 256，position embedding 可能需要二维插值，不能简单截断或复制。

### 8.4 把 class token 用于分割

分类 token 是整图摘要，分割需要 patch-level token 和空间还原。

### 8.5 显存不足

减少 batch、输入分辨率、token 数或使用 patch merging、窗口注意力和混合精度。详见 [[Transformer]] 和 [[PyTorch]]。

### 8.6 预训练输入分布不一致

检查 resize、crop、normalize 和颜色顺序，见 [[图像基础]]。

## 9. 容易混淆

| 概念 | 含义 | 不要混淆 |
|---|---|---|
| patch | 图像小块 | 一个类别 |
| token | patch 的向量表示 | 原始像素块 |
| N | patch/token 数 | batch |
| D | token embedding 维度 | 图像宽度 |
| class token | 整图摘要 | 每个 patch |
| position embedding | 位置表示 | 内容特征 |
| patch size | 每块像素边长 | token 维度 |

## 10. 与其他笔记的关系

前置：[[图像基础]]、[[Transformer]]、[[Attention]]、[[PyTorch]]。

相关：[[CNN视觉模型]]、[[视觉表征]]、[[视频理解]]、[[自监督学习]]。

本篇不展开完整 ViT 论文变体、检测器和视频 Transformer 的全部实现。

## 11. 知识地图

```text
图像 [B,C,H,W]
↓
patch embedding
↓
[B,N,D]
↓
class token / position
↓
Transformer blocks
↓
全局或空间 token
↓
视觉任务 head
↓
评估与部署
```

## 12. 快速查询表

| 检查项 | 计算/问题 |
|---|---|
| patch 数 | (H/P)×(W/P) |
| token shape | [B,N,D] |
| class token | N 是否加 1 |
| position | 是否与 N 匹配 |
| dense task | 是否保留 patch token |
| 部署 | token 数、显存、延迟是否可接受 |

## 13. 练习题（题目下附参考答案）

1. ViT 把图像转换成什么？

**参考答案：** 转换成 patch token 序列，再交给 Transformer 处理。

2. 224×224 图像、patch size=16 时有多少 patch？

**参考答案：** 每边 14 个，共 196 个。

3. patch embedding 的作用是什么？

**参考答案：** 把每个图像 patch 的像素表示映射到 D 维 token。

4. 为什么需要 position embedding？

**参考答案：** Attention 本身缺少天然空间顺序，需要位置表示区分 patch 的位置。

5. class token 用来做什么？

**参考答案：** 汇聚整图信息，常用于图像分类；具体实现也可能使用平均池化。

6. 输入 [8,3,224,224]、patch=16、D=768，token shape 是什么？

**参考答案：** 不含 class token 时为 [8,196,768]，含 class token 时为 [8,197,768]。

7. Conv2d patch embedding 输出 [8,768,14,14] 后如何变为 token？

**参考答案：** flatten 空间维并转置，得到 [8,196,768]。

8. ViT 与 CNN 的主要先验差异是什么？

**参考答案：** CNN 强调局部连接和参数共享，ViT 通过 token attention 建模关系。

9. 为什么 patch 越小计算可能越大？

**参考答案：** patch 越小，token 数越多，而 attention 对序列长度的成本通常近似平方增长。

10. 分割任务为什么不能只使用 class token？

**参考答案：** class token 是整图摘要，分割需要每个空间位置的 patch token。

11. 改变输入分辨率要检查什么？

**参考答案：** patch 数、position embedding、预处理和显存。

12. ViT 预训练迁移时为什么要检查 normalize？

**参考答案：** 输入分布与预训练不匹配会削弱预训练表示。

13. 机器人闭环中 ViT 还要评估什么？

**参考答案：** 延迟、显存、帧率、空间定位和闭环成功率，而不仅是离线准确率。

14. position embedding shape 不匹配怎么办？

**参考答案：** 计算当前 patch 网格，按实现进行二维插值或调整输入分辨率，不能盲目 reshape。

15. patch token 与 class token 的用途有什么不同？

**参考答案：** patch token 保留空间局部信息，class token 汇聚整图信息。

16. 为什么 ViT 常需要大规模预训练？

**参考答案：** 它的局部视觉先验较弱，小数据上更容易过拟合或学不到稳定结构。

17. 多相机输入如何进入 ViT？

**参考答案：** 可将各相机 patch token 拼接并加入相机/位置标识，但需处理 token 数与视角语义。

18. World Model 中 ViT encoder 可能输出什么？

**参考答案：** 图像 patch 表示或压缩视觉 latent，供动力学预测和规划使用。

19. 研究代码中 token 数突然变多，怎么查？

**参考答案：** 检查输入分辨率、patch size、padding、class token、多个视角拼接和时间维展开。

20. 为什么不能只复制 CNN 的下采样配置到 ViT？

**参考答案：** ViT 的 token 化、attention 成本、位置编码和空间任务要求不同。

## 13. 掌握度检查（附答案）

1. patch 是什么？

**答案：** 图像中的固定大小小块。

2. token 是什么？

**答案：** patch 经 embedding 后得到的向量表示。

3. patch 数如何计算？

**答案：** 在可整除时为 (H/P)×(W/P)，再根据 class token 等机制调整序列长度。

4. token shape [B,N,D] 如何解释？

**答案：** batch 数、token 数、每个 token 的 embedding 维度。

5. 为什么需要位置编码？

**答案：** 让模型知道每个 patch 的空间位置。

6. class token 与 patch token 的区别是什么？

**答案：** 前者常做整图摘要，后者保留局部空间内容。

7. ViT 为什么可能显存昂贵？

**答案：** patch 变成较长序列，attention 的配对成本随 token 数平方增长。

8. ViT 与 CNN 的差异是什么？

**答案：** CNN 有强局部空间先验，ViT 用 attention 学习 patch 间关系。

9. 为什么分割需要空间 token？

**答案：** 每个像素或区域都需要对应的空间预测。

10. 输入分辨率变化会影响什么？

**答案：** patch 数、位置编码、显存和模型输出空间对应关系。

11. ViT 在具身智能中可能有什么优势？

**答案：** 能融合远距离视觉区域、多视角和语言/状态 token。

12. ViT 在具身智能中可能有什么风险？

**答案：** 延迟、显存、数据需求和小目标/空间精度问题。

13. 何时查 [[Transformer]]？

**答案：** 需要理解 block、mask、FFN、残差和位置编码实现时。

14. 何时查 [[图像基础]]？

**答案：** 需要检查 RGB、resize、normalize、HWC/CHW 时。

15. 阅读 ViT 代码最先看什么？

**答案：** patch embedding、token shape、position embedding、class token 和输出 head。
