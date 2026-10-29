# CNN：让模型理解图像中的局部与空间结构

卷积神经网络（Convolutional Neural Network，CNN）是处理图像和空间网格数据的经典结构。它利用局部连接、参数共享和逐步下采样，从像素学习边缘、纹理、物体部件和更高层视觉特征。

本文重点负责卷积、通道、空间尺寸、池化和视觉 backbone；训练目标、优化器和评估分别见 [[损失函数]]、[[优化器]]、[[模型评估]]。CNN 输出的中间表示与 [[表征学习]] 相连。

## 0. 学习目标

学完以后能够：

- 解释为什么图像不能简单当成长向量处理；
- 读懂 Conv2d 的输入输出 shape 和主要参数；
- 计算卷积后空间尺寸；
- 理解通道、感受野、池化、stride 和 padding；
- 读懂 CNN backbone 到分类头或动作头的流程；
- 排查图像通道顺序、shape、尺寸和显存错误。

## 1. 为什么需要 CNN？

一张 RGB 图片可以表示为高度 H、宽度 W、通道 C 的网格。相邻像素通常有局部关系。若把所有像素直接交给一个巨大 Linear：

- 参数量会随图片尺寸急剧增加；
- 同一个边缘出现在不同位置时需要重复学习；
- 模型不容易利用局部空间结构。

CNN 引入一个小滤波器，在图像不同位置重复使用：

```text
图像
↓
局部卷积核扫描
↓
特征图
↓
下采样
↓
更大感受野、更抽象特征
↓
分类 / 检测 / 动作预测
```

## 2. 核心概念

### 2.1 通道（Channel）

通道是同一空间位置上的不同数值平面。RGB 图片有 3 个输入通道；CNN 中间层的 64 个通道可以理解为 64 种学习到的特征响应。

PyTorch 的 Conv2d 通常使用 NCHW：

```text
[N, C, H, W]
N：batch
C：channel
H：height
W：width
```

### 2.2 卷积层（Convolution Layer）

在深度学习代码中，Conv2d 通常实现的是交叉相关，但大家仍习惯称为卷积。它用可学习的 kernel 在局部窗口上计算加权和。

```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1,
)
x = torch.randn(8, 3, 64, 64)
y = conv(x)
print(y.shape)  # [8, 32, 64, 64]
```

3 是输入通道，32 是输出通道，3 是核的高宽，stride=1 表示每次移动一个像素，padding=1 在边缘补一圈，使空间尺寸保持 64。

### 2.3 输出通道

out_channels 不是输出图片的宽高，而是输出特征图的数量。输出 shape 是 [N, out_channels, H_out, W_out]。

### 2.4 卷积输出尺寸

当 kernel_size、stride、padding 为整数且 dilation=1 时：

$$
H_{out} = \lfloor (H+2P-K)/S \rfloor + 1
$$

宽度同理。

例如 H=32，K=3，P=1，S=2：

$$
\lfloor(32+2-3)/2\rfloor+1=16
$$

### 2.5 感受野（Receptive Field）

感受野是某个高层特征对应原图的区域。堆叠卷积和下采样后，一个单元可以看到更大的图像区域。早期层看边缘，后期层可能看物体部件或整体结构。

### 2.6 池化与 stride

Pooling 用固定规则汇聚邻域，例如 MaxPool 取最大值，AvgPool 取平均值。stride 大于 1 的卷积也可以下采样。

下采样减少空间尺寸和计算量，但会丢失部分细节；机器人视觉任务中要权衡精度、速度和控制所需的空间信息。

## 3. CNN 的整体工作原理

```text
[N,3,H,W]
↓
Conv：局部模式 + 通道变换
↓
Activation：非线性
↓
Conv / Pool：更大感受野
↓
Feature map
↓
Global Average Pooling 或 Flatten
↓
MLP / task head
↓
prediction
```

CNN 的中间 feature map 不是最终类别，而是可供下游使用的视觉表示。

## 4. 从零读懂一个 CNN

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)
```

若输入 [N,3,64,64]：

- 第一层后 [N,32,64,64]；
- MaxPool 后 [N,32,32,32]；
- 第二层后 [N,64,32,32]；
- AdaptiveAvgPool 后 [N,64,1,1]；
- flatten(1) 后 [N,64]；
- head 后 [N,num_classes]。

### 4.1 flatten(1) 为什么从 1 开始？

第 0 维通常是 batch。flatten(1) 把通道、高度、宽度合并，不破坏 batch。若使用 flatten()，可能连 batch 也合并，造成严重错误。

## 5. 图像预处理与 shape

科研代码中常见流程：

```text
磁盘图像 HWC、uint8、0~255
↓
读取与增强
↓
转换为 float32
↓
归一化
↓
变为 CHW
↓
DataLoader 组成 NCHW
↓
送进 CNN
```

检查：

```python
print(images.shape, images.dtype)
print(images.min().item(), images.max().item())
print(torch.cuda.is_available())
```

RGB 图像若错误地把 HWC 当成 CHW，通常会出现通道数或空间尺寸错误。

## 6. 在 AI 与具身智能中的使用

CNN 常见于：

- 视觉分类、检测和分割；
- 机器人相机图像编码；
- 行为克隆的视觉 backbone；
- 世界模型的 observation encoder；
- 视频模型的每帧空间编码；
- 视觉特征与 proprioception 的融合。

一个机器人策略可能是：

```text
camera image [B,3,H,W]
↓
CNN encoder
↓
visual latent [B,D]
↓
与 proprioceptive state 拼接
↓
MLP policy
↓
action
```

图像增强是否改变标签要参考 [[正则化]]；视觉表示是否可迁移见 [[表征学习]]。

## 7. 常见错误与 Debug

### 7.1 NCHW 与 NHWC 混淆

现象：expected input to have 3 channels，或输出 shape 异常。

判断：打印 shape；PyTorch Conv2d 通常需要 [N,C,H,W]。如果数据是 [N,H,W,C]，需要可靠地 permute，而不是 reshape。

### 7.2 Conv2d 输入通道错误

第一层 in_channels 必须与输入 C 一致。灰度图通常 C=1，RGB 通常 C=3。

### 7.3 Linear 维度接不上

不要手算复杂下采样后直接写死尺寸；可使用 AdaptiveAvgPool2d，或先用假输入运行一遍并打印 shape。

### 7.4 图像数值范围错误

uint8 0~255、float 0~1、标准化后的数值含义不同。错误归一化会让训练不稳定或预训练模型失效。

### 7.5 过度下采样

空间尺寸太早变成 1×1，可能丢失抓取、边界和小物体信息。具身任务要根据控制需要保留细节。

### 7.6 显存不足

输入分辨率、batch、通道数和中间 feature map 都会影响显存。先打印 shape，减小 batch 或分辨率，再考虑混合精度和梯度累积，详见 [[PyTorch]]。

## 8. 容易混淆的概念

| 概念 | 含义 | 常见误解 |
|---|---|---|
| channel | 特征平面数量 | 图片宽度 |
| kernel | 局部可学习权重 | 整张图片 |
| stride | 滑动步长 | batch size |
| padding | 边缘填充 | 数据增强 |
| pooling | 固定汇聚 | 可学习卷积 |
| feature map | 中间空间特征 | 最终类别 |
| receptive field | 对应原图区域 | 卷积核尺寸本身 |

## 9. 与其他笔记的关系

前置：[[PyTorch]]、[[监督学习]]、[[损失函数]]、[[正则化]]、[[模型评估]]。

相关：[[MLP]]、[[Attention]]、[[Transformer]]、[[表征学习]]、[[自监督学习]]。

本篇不展开检测、分割和视频时序建模；只说明 CNN 作为视觉 backbone 的职责。

## 10. 知识地图

```text
图像 HWC
↓
预处理与归一化
↓
NCHW
↓
Conv + Activation
↓
Pool / Stride
↓
高层 feature map
↓
Pooling / Flatten
↓
MLP 或其他 head
↓
任务评估
```

## 11. 快速查询表

| 任务 | 重点 |
|---|---|
| 查看 CNN 输入 | NCHW、dtype、数值范围 |
| 计算空间尺寸 | H_out = floor((H+2P-K)/S)+1 |
| 接 Linear | 先确认 flatten 后最后一维 |
| 接预训练模型 | 检查输入归一化、通道数和 eval |
| 视觉策略 | CNN latent 与状态拼接前检查 shape |

## 12. 练习题（题目下附参考答案）

1. CNN 为什么适合图像？

**参考答案：** 图像有局部空间结构，CNN 通过局部连接和参数共享减少参数并复用相同模式。

2. PyTorch Conv2d 通常接收什么 shape？

**参考答案：** [N,C,H,W]。

3. out_channels 表示什么？

**参考答案：** 输出特征图的数量，不是输出高度或宽度。

4. padding 的主要作用是什么？

**参考答案：** 在边缘补值，控制空间尺寸并减少边界信息过快丢失。

5. stride=2 会怎样？

**参考答案：** 卷积窗口每次移动两格，通常使空间尺寸约减半。

6. 卷积核越大一定越好吗？

**参考答案：** 不一定。核大可能扩大局部范围但增加计算和参数，也可能损失细节。

7. 输入 [8,3,64,64] 经过 Conv2d(3,32,3,padding=1) stride=1 后 shape 是什么？

**参考答案：** [8,32,64,64]。

8. 上题后接 MaxPool2d(2)，shape 是什么？

**参考答案：** [8,32,32,32]。

9. 为什么 flatten(1) 而不是 flatten()？

**参考答案：** 保留 batch 维，把后面的通道和空间维合并。

10. 输入是 NHWC，能直接送 Conv2d 吗？

**参考答案：** 通常不能。应检查并正确转换为 NCHW，不能用 reshape 冒充维度交换。

11. CNN 为什么常作为机器人视觉 encoder？

**参考答案：** 它能把高维图像变成包含局部到高层空间信息的紧凑表示，便于与状态融合并预测动作。

12. 如何让 CNN 分类头接上 feature map？

**参考答案：** 先用全局池化或正确 flatten 得到 [N,D]，再用 Linear(D,num_classes)。

13. Conv2d 报输入通道错误，检查什么？

**参考答案：** 图像真实 C、第一层 in_channels、颜色通道处理和是否误把 HWC 当 NCHW。

14. 图像输入数值是 0~255，模型预期 0~1，会怎样？

**参考答案：** 输入尺度错误，可能导致激活和训练不稳定；应按模型预期转换和归一化。

15. 验证预处理可以使用随机裁剪吗？

**参考答案：** 通常验证要稳定、可重复，不能随意使用训练时的随机增强，除非明确设计测试时增强。

16. 训练 CNN 显存不足时排查顺序是什么？

**参考答案：** 查看输入分辨率和 batch，再看 feature map 与模型宽度；先减小 batch/分辨率，之后再用混合精度等方法。

17. 研究代码中 CNN 输出用于策略，为什么不能只评估分类准确率？

**参考答案：** 分类准确率不一定代表动作和闭环控制有效，应评估下游策略或任务成功率。

18. CNN 下采样太多会影响什么？

**参考答案：** 细小物体、边界、抓取点和空间定位信息可能丢失。

19. 预训练 CNN 迁移到机器人相机时要检查什么？

**参考答案：** 通道、分辨率、归一化、传感器分布、冻结策略和训练/评估模式。

20. CNN feature 与 proprioception 拼接前必须确认什么？

**参考答案：** 两者 batch 维一致、feature 维语义明确、dtype/device 一致，并确认拼接维度。

## 13. 掌握度检查（附答案）

1. CNN 的参数共享解决了什么问题？

**答案：** 同一种局部模式可以在不同位置复用同一组卷积参数，减少参数并利用平移相关性。

2. N、C、H、W 分别是什么？

**答案：** batch、通道、高度、宽度。

3. out_channels 是否等于图片宽度？

**答案：** 不等于，它是输出特征图数量。

4. padding 和 stride 分别影响什么？

**答案：** padding 影响边缘与空间尺寸，stride 影响窗口移动步长和下采样程度。

5. 感受野是什么？

**答案：** 某个高层特征对应原图的区域范围。

6. 为什么需要池化或下采样？

**答案：** 减少空间尺寸和计算量，扩大后续单元的有效感受野，但会损失细节。

7. CNN 输出 feature map 后为什么还要 pooling 或 flatten？

**答案：** 将空间特征转换为适合分类、动作或其他 head 的向量表示。

8. 如何计算卷积输出空间尺寸？

**答案：** 使用 floor((H+2P-K)/S)+1，并对宽度同样计算。

9. HWC 与 NCHW 的区别是什么？

**答案：** 前者把通道放最后，后者把 batch 和通道放在前；PyTorch Conv2d 通常用 NCHW。

10. 图像归一化为什么重要？

**答案：** 它影响输入尺度、训练稳定性以及预训练权重的输入分布匹配。

11. CNN 在具身智能中通常输出什么？

**答案：** 视觉 latent 或 feature，供策略、价值、世界模型或规划模块使用。

12. 为什么不应盲目把 feature map reshape？

**答案：** reshape 不会真正交换语义维度，可能制造看似正确但语义错误的数据。

13. CNN 与 MLP 的主要差别是什么？

**答案：** CNN 利用局部空间结构和共享卷积核，MLP 通常对特征做全连接变换。

14. 何时查正则化笔记？

**答案：** 需要处理视觉过拟合、数据增强、Dropout、权重衰减或早停时。

15. 阅读 CNN research code 最先看什么？

**答案：** 输入格式、每层 shape、下采样位置、输出 feature 的用途和后接评估指标。
