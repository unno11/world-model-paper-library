# CNN视觉模型：从视觉骨干到分类、检测与分割

CNN 视觉模型不是只有一层 Conv2d，而是一套从图像输入到视觉任务输出的结构。它通常包含 backbone、neck、head 和 loss。理解这些职责，才能读懂 ResNet、U-Net、FPN、检测器以及机器人视觉策略代码。

通用卷积数学和基础 shape 已在 [[CNN]] 介绍；本文重点负责视觉任务架构、预训练 backbone、分类/检测/分割输出以及研究代码阅读。

## 0. 学习目标

学完以后能够：

- 区分 backbone、neck、head、feature map 和 task output；
- 看懂图像分类、目标检测、语义/实例分割的基本输出；
- 理解 ResNet 的残差块、U-Net 的 skip connection 和多尺度特征；
- 读懂预训练视觉模型的冻结、替换 head 和微调；
- 判断视觉任务 loss、标签和评估指标是否匹配；
- 将 CNN 视觉模型连接到机器人感知与控制。

## 1. 为什么需要不同的 CNN 视觉模型？

不同任务需要不同输出：

| 任务 | 问题 | 输出 |
|---|---|---|
| 分类 | 图中是什么 | 类别 logits |
| 检测 | 有什么、在哪里 | 框 + 类别 |
| 语义分割 | 每个像素属于什么类 | H×W mask |
| 实例分割 | 每个物体实例的像素 | 每实例 mask |
| 深度估计 | 每个像素的距离 | 深度图 |
| 视觉策略 | 下一步做什么 | 动作或价值 |

单个全局向量适合分类，但不适合精确定位，因此视觉模型要保留或恢复空间结构。

## 2. 组件职责

### 2.1 Backbone

backbone 是提取通用视觉特征的主体，例如 ResNet、ConvNeXt。它把图像变为一层或多层 feature maps。

### 2.2 Neck

neck 融合不同层级的特征，常见于 FPN。浅层分辨率高、细节多；深层分辨率低、语义强。

### 2.3 Head

head 将 feature maps 转成特定任务输出。分类 head 可输出类别，检测 head 输出框和类别，分割 head 输出像素级预测。

### 2.4 Residual Block

ResNet 残差块学习 F(x)+x，而不是直接学习完整映射。skip connection 提供更直接的梯度和信息路径。

```text
x
├───────────────┐
↓               │
卷积变换 F(x)    │
↓               │
F(x) + x  ←─────┘
```

相加前 shape 必须一致；通道或空间尺寸变化时需要 projection shortcut。

### 2.5 U-Net Skip Connection

U-Net 编码器逐步下采样获取语义，解码器逐步上采样恢复空间细节，并通过 skip connection 融合浅层特征。

```text
encoder 高分辨率细节 ─────┐
       ↓                  │
encoder 低分辨率语义 → decoder 上采样 → 像素输出
```

## 3. 视觉任务输出

### 3.1 分类

输入 [B,3,H,W]，输出 [B,K] 的 logits。K 是类别数。使用 [[损失函数]] 中的分类目标和 [[模型评估]] 中的分类指标。

### 3.2 检测

每张图有若干目标，输出通常包含：

- bounding box 坐标；
- objectness 或置信度；
- 类别分数。

目标数量可变，因此数据集的 collate 和标签格式比分类复杂。

### 3.3 语义分割

输出 [B,K,H,W] 的每像素类别 logits。目标通常是 [B,H,W] 的类别索引，或者适合二元/多标签的 mask。

### 3.4 深度与姿态

深度估计输出连续值图；关键点或姿态估计输出位置坐标或热图。必须根据目标的几何含义选择 loss 和指标。

## 4. 多尺度特征

物体大小变化很大。FPN 等结构利用不同分辨率特征：

```text
浅层：高分辨率，细节丰富，语义较弱
中层：平衡
深层：低分辨率，语义强，细节少
↓
多尺度融合
↓
小目标和大目标都获得合适特征
```

机器人抓取中的小物体、远处目标和近距离大物体都可能需要多尺度信息。

## 5. 从零读懂迁移学习

```python
backbone = torchvision.models.resnet18(weights="DEFAULT")
in_dim = backbone.fc.in_features
backbone.fc = nn.Linear(in_dim, num_classes)

for name, p in backbone.named_parameters():
    if not name.startswith("fc."):
        p.requires_grad = False
```

这里先加载预训练 backbone，再替换分类 head。冻结策略要与 optimizer 参数、BatchNorm 行为和数据规模一起检查。下游输入预处理必须与预训练期望匹配，见 [[图像基础]]。

## 6. 视觉模型训练流程

```text
图像与标签
↓
backbone / neck
↓
task head
↓
分类、框、mask、深度或动作 loss
↓
backward
↓
optimizer.step
↓
验证指标与可视化
↓
checkpoint
```

检测和分割不能只打印一个 loss 就判断模型好坏，要可视化框/mask 并看任务指标。

## 7. 在具身智能中的使用

CNN 视觉模型可能用于：

- 物体检测和抓取目标定位；
- 语义分割和可通行区域估计；
- 深度、姿态和空间占据预测；
- 视觉 observation encoder；
- 视觉特征与 proprioception 融合；
- 视觉输入到 policy 或 World Model。

典型路径：

```text
camera RGB/depth
↓
CNN backbone
↓
spatial feature maps
├─ detection head → object boxes
├─ segmentation head → pixel masks
└─ pooling → visual latent → policy
```

## 8. 常见错误与 Debug

### 8.1 替换 head 忘记更新类别数

模型输出类别数仍是预训练数据集的类别数，标签却是新任务类别数。打印最后输出 shape。

### 8.2 分割标签 dtype 错

多类 CrossEntropyLoss 常需要 Long 类别索引，不是 one-hot float；二元分割的 loss 约定不同，要看具体实现。

### 8.3 检测框坐标系错误

xyxy、xywh、归一化坐标和像素坐标混用会导致框完全错位。可视化一批真实标签和预测。

### 8.4 下采样太多

小目标或抓取点在低分辨率 feature 中消失。检查各 stage shape 和 stride。

### 8.5 只冻结参数但忘记 BatchNorm 模式

冻结 weight 不等于运行行为完全固定。训练/评估模式会影响 BatchNorm running statistics，需按实验意图设置。

### 8.6 只看 loss 不看空间输出

视觉任务中的 loss 下降不保证框和 mask 合理。始终抽样可视化。

## 9. 容易混淆

| 概念 | 含义 | 不等于 |
|---|---|---|
| backbone | 通用特征提取 | 完整任务模型 |
| neck | 多层特征融合 | 分类标签 |
| head | 任务输出模块 | backbone 参数 |
| feature map | 有空间结构的中间特征 | 最终预测 |
| detection | 框 + 类别 | 只有分类 |
| semantic mask | 每像素类别 | 每个实例独立 mask |
| freeze | 不更新参数 | 必然完全 eval |
| pretrain | 先学通用任务 | 一定适合下游 |

## 10. 与其他笔记的关系

前置：[[图像基础]]、[[CNN]]、[[PyTorch]]、[[损失函数]]、[[模型评估]]、[[正则化]]。

相关：[[ViT]]、[[视觉表征]]、[[视频理解]]。

本篇不展开具体检测器的全部工程实现；重点是读懂视觉模型组件和输出职责。

## 11. 知识地图

```text
图像
↓
backbone 提取特征
↓
neck 融合尺度
↓
head 适配任务
├─ 分类
├─ 检测
├─ 分割
├─ 深度/姿态
└─ 视觉策略
↓
任务 loss + 可视化 + 指标
```

## 12. 快速查询表

| 看到的模块 | 先问什么 |
|---|---|
| backbone | 输出哪些 feature map |
| neck | 融合哪些尺度 |
| head | 输出框、mask、类别还是动作 |
| freeze | 哪些参数训练，BatchNorm 怎么办 |
| loss | 目标 shape/dtype 是否匹配 |
| eval | 指标和可视化是否同时检查 |

## 13. 练习题（题目下附参考答案）

1. backbone 的职责是什么？

**参考答案：** 从图像提取通用视觉特征，通常不直接决定最终任务输出。

2. head 的职责是什么？

**参考答案：** 把特征转换为具体任务输出，如类别、框、mask 或动作。

3. 分类与检测的输出区别是什么？

**参考答案：** 分类通常输出整图类别，检测还要输出目标位置和每个目标的类别/置信度。

4. U-Net 的 skip connection 为什么有用？

**参考答案：** 把编码器的高分辨率细节传给解码器，帮助恢复像素级空间信息。

5. FPN 为什么使用多尺度特征？

**参考答案：** 不同尺度特征分别适合细节和语义，融合后更适合大小不同的目标。

6. ResNet 残差块的基本形式是什么？

**参考答案：** F(x)+x；需要相加的两路 shape 一致。

7. 分类输出 [B,K] 中 K 是什么？

**参考答案：** 类别数量。

8. 多类分割输出通常是什么 shape？

**参考答案：** [B,K,H,W] 的像素类别 logits。

9. 替换预训练 head 时必须检查什么？

**参考答案：** 新任务类别数、head 输入维度、标签格式、optimizer 参数和输入归一化。

10. 为什么检测框标签需要可视化？

**参考答案：** 坐标格式和缩放错误可能不报错，但会导致框位置完全错误。

11. 小目标在 CNN 中为什么容易丢失？

**参考答案：** 多次 stride/pooling 下采样会使小目标信息在低分辨率 feature map 中消失。

12. 分割标签为 one-hot float 能直接交给所有 loss 吗？

**参考答案：** 不能。不同 loss 对 shape/dtype 要求不同，多类 CE 常用 Long 类别索引。

13. 冻结 backbone 后为什么还要关注 BatchNorm？

**参考答案：** train/eval 会影响 running statistics，参数冻结不代表行为完全固定。

14. 视觉任务为什么不能只看 loss？

**参考答案：** 框、mask、深度和动作有空间/任务语义，需结合可视化和任务指标。

15. 视觉策略如何使用 CNN 模型？

**参考答案：** backbone 生成视觉 latent，与机器人状态融合后交给 policy head 输出动作。

16. 预训练 ResNet 输入归一化不匹配会怎样？

**参考答案：** 输入分布与预训练权重不匹配，迁移效果可能显著下降。

17. 目标框是 xywh，代码当 xyxy 解析，如何判断？

**参考答案：** 可视化标签框，检查坐标关系和数据集格式定义，不要只看数值范围。

18. 检测模型离线指标高但抓取失败，可能是什么？

**参考答案：** 评估场景与真实相机分布不同、框精度不足、坐标变换错误或闭环控制没有评估。

19. 为什么机器人视觉模型可能需要深度或分割 head？

**参考答案：** 空间距离、可通行区域、目标轮廓等结构信息可帮助定位和控制。

20. 阅读 CNN 视觉 repository 的顺序是什么？

**参考答案：** 先看 dataset 和标签，再看 backbone/neck/head，最后看 loss、评估和可视化。

## 13. 掌握度检查（附答案）

1. backbone、neck、head 的关系是什么？

**答案：** backbone 提取特征，neck 融合特征，head 产生任务输出。

2. 为什么分类头不适合直接做分割？

**答案：** 分类头通常只保留全局向量，分割需要保留或恢复每个像素的空间信息。

3. 残差连接解决什么问题？

**答案：** 提供直接信息与梯度路径，帮助深层网络训练。

4. U-Net 的编码器和解码器分别做什么？

**答案：** 编码器下采样提取语义，解码器上采样恢复空间细节。

5. 多尺度特征有什么价值？

**答案：** 同时处理小目标细节和大目标语义。

6. 分类 logits 与分割 logits 的 shape 差别是什么？

**答案：** 分类常为 [B,K]，分割常为 [B,K,H,W]。

7. 为什么必须检查标签 dtype？

**答案：** loss 对类别索引、one-hot、连续值的 shape/dtype 要求不同。

8. freeze 与 eval 是否相同？

**答案：** 不同。freeze 控制参数梯度，eval 控制模块行为如 Dropout/BatchNorm。

9. 视觉任务为什么要可视化？

**答案：** 发现坐标、颜色、空间和输出语义错误，这些错误未必由 loss 暴露。

10. 小目标受什么影响？

**答案：** 输入分辨率、下采样 stride、feature map 尺度和多尺度融合。

11. CNN 模型在机器人中输出什么？

**答案：** 视觉特征、目标框、mask、深度、姿态或动作相关表示。

12. 预训练迁移首先要匹配什么？

**答案：** 输入通道、分辨率、归一化、输出 head 和任务标签。

13. 为什么检测标签不能只看类别？

**答案：** 检测还依赖位置，坐标格式和几何变换必须正确。

14. 何时查 [[图像基础]]？

**答案：** 需要排查图像 shape、颜色、dtype、归一化和标签坐标时。

15. 读视觉模型代码最先找什么？

**答案：** 数据和标签格式、backbone 输出、head 输出、loss 目标与评估可视化。
