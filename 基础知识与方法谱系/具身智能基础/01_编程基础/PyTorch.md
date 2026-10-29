# PyTorch：从 Tensor 到 AI 训练代码

PyTorch 是许多深度学习、计算机视觉、机器人学习、World Model 和具身智能项目使用的 Python 框架。它不只是一个“把数组放到 GPU 上”的库，而是一套连接数据、模型、自动微分、优化器和训练流程的工具。

真实 AI 项目通常形成这样的数据流：

```text
原始数据
    ↓
Dataset / DataLoader
    ↓
batch Tensor
    ↓
model.forward
    ↓
预测 outputs
    ↓
loss
    ↓
autograd 计算梯度
    ↓
optimizer 更新参数
    ↓
checkpoint / evaluation
```

本篇的目标是让你能读懂这条链路，知道每个对象为什么存在、输入输出是什么，以及报错时应该检查哪里。

> [!NOTE]
> 本篇负责 Tensor、shape、dtype、device、autograd、nn.Module、Dataset、DataLoader、loss、optimizer、train/eval、checkpoint 和基础 GPU 使用。Python 语法见 [[Python基础]]；NumPy 数组见 [[NumPy]]；环境和包安装见 [[Conda与环境管理]]；命令行见 [[Linux与Shell]]；通用排错见 [[Debug与工程基础]]。

## 0. 学习目标

学完以后，应该能够：

- 解释 Tensor、shape、dtype、device 和 batch。
- 创建 Tensor，查看和转换 Tensor 的状态。
- 理解 NumPy array 与 PyTorch Tensor 的关系。
- 理解 requires_grad、计算图、loss 和 backward。
- 读懂 nn.Module、参数、forward 和子模块。
- 理解 Dataset 和 DataLoader 如何提供 batch。
- 读懂 optimizer、zero_grad、backward 和 step 的训练顺序。
- 区分 model.train()、model.eval() 和 torch.no_grad()。
- 处理 CPU/GPU、dtype、shape 和显存问题。
- 保存和加载 checkpoint，并知道 checkpoint 不能代替代码、环境和配置。
- 按最小路径 Debug：一个样本 → 一个 batch → forward → loss → backward → update。

## 1. 为什么需要 PyTorch？

Python 列表或 NumPy 数组可以保存数字，但完整深度学习训练还需要：

1. 规则的多维数值结构。
2. 高效的 CPU/GPU 运算。
3. 根据 loss 自动计算参数梯度。
4. 定义由许多层组成的模型。
5. 按 batch 读取数据。
6. 更新模型参数并保存状态。

PyTorch 把这些能力组合起来：

```text
Tensor：保存数据
    ↓
nn.Module：组织模型计算
    ↓
autograd：计算梯度
    ↓
Optimizer：根据梯度更新参数
    ↓
训练循环：重复上述过程
```

如果只把 Tensor 当作“更快的 NumPy 数组”，就会漏掉 autograd、模型参数、训练模式和 checkpoint 等核心概念。

## 2. 核心概念

### 2.1 Tensor（张量）

**中文名称**：张量。

**英文名称**：Tensor。代码和论文中通常直接写 Tensor 或 tensor。

**一句话直觉**：Tensor 是适合深度学习计算的多维数值容器。

**正式含义**：Tensor 保存有规则 shape 的数值，并携带 dtype、device 等信息；它可以参与 PyTorch 运算，也可以记录计算图以便自动微分。

从维度看：

```python
import torch

scalar = torch.tensor(3.0)              # 0 维
vector = torch.tensor([1.0, 2.0])       # 1 维
matrix = torch.zeros(2, 3)              # 2 维
volume = torch.zeros(2, 3, 4)           # 3 维
```

Tensor 不是“神奇的模型”，也不自动代表图像、状态或动作。每一维的含义必须由项目代码决定。

### 2.2 shape

shape 表示 Tensor 每个维度的长度。

```python
x = torch.randn(32, 128)
print(x.shape)
print(x.ndim)
print(x.numel())
```

这里：

- 32 通常可以表示 batch size。
- 128 可以表示每个样本的特征维度。
- shape 是 (32, 128)。
- ndim 是 2。
- numel 是 4096。

shape 的数字本身没有固定含义。视觉项目可能使用 B,C,H,W；轨迹项目可能使用 B,T,D；文本项目可能使用 B,T。必须看数据流和模型接口。

### 2.3 dtype

dtype 表示 Tensor 中元素的数据类型。

```python
x = torch.tensor([1, 2, 3], dtype=torch.int64)
y = torch.tensor([1.0, 2.0], dtype=torch.float32)

print(x.dtype)
print(y.dtype)
```

常见 dtype：

- torch.float32：常见模型计算类型。
- torch.float64：更高精度但通常更耗内存。
- torch.float16 / bfloat16：混合精度和加速场景常见。
- torch.int64：索引和部分分类标签常见。
- torch.bool：掩码。

dtype 不只是格式问题。分类损失、矩阵运算和模型参数都可能要求特定 dtype。遇到错误先打印 dtype，而不是盲目转换。

### 2.4 device

device 表示 Tensor 位于 CPU 还是某个 GPU。

```python
x = torch.zeros(2, 3)
print(x.device)

if torch.cuda.is_available():
    x = x.to("cuda")
    print(x.device)
```

模型参数和参与同一计算的 Tensor 通常需要在兼容设备上。常见错误：

```text
Expected all tensors to be on the same device
```

这通常表示一部分对象在 CPU，另一部分在 GPU。

### 2.5 batch

batch 是一次同时处理的一组样本。

```python
x = torch.randn(32, 128)
```

如果 x 表示一批特征，32 是 batch size，128 是每个样本的特征数。batch size 影响：

- 单次计算的样本数。
- 显存使用。
- 梯度估计的统计性质。
- 训练速度和可能的训练动态。

batch 不是“随便加的一维”。如果把单个样本误当成 batch，模型可能收到错误 shape。

## 3. 创建与检查 Tensor

### 3.1 常见创建方式

```python
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random = torch.randn(2, 3)
integers = torch.arange(6)
full = torch.full((2, 3), 7.0)
```

randn 从标准正态分布生成随机数；arange 生成整数序列。创建后要检查 dtype 和 shape。

```python
x = torch.randn(32, 128, dtype=torch.float32)
print(x.shape, x.dtype, x.device)
```

### 3.2 从 Python 和 NumPy 转换

```python
import numpy as np
import torch

array = np.ones((2, 3), dtype=np.float32)
tensor_a = torch.from_numpy(array)
tensor_b = torch.tensor(array)

print(tensor_a.shape, tensor_a.dtype)
print(tensor_b.shape, tensor_b.dtype)
```

from_numpy 常与 NumPy 数组共享内存；torch.tensor 通常创建独立 Tensor。需要确认共享关系时进行小实验。NumPy 细节见 [[NumPy]]。

### 3.3 Tensor 转 NumPy

```python
x = torch.ones(2, 3)
array = x.numpy()
```

只有位于 CPU、且不需要梯度的 Tensor 才能直接安全地转换。训练中常见写法：

```python
array = tensor.detach().cpu().numpy()
```

含义：

- detach：不再沿当前计算图追踪。
- cpu：把 Tensor 移到 CPU。
- numpy：转换为 NumPy 数组。

不要在模型仍需梯度时随意转 NumPy，因为这会离开 PyTorch 的自动微分流程。

### 3.4 检查数值

```python
def inspect_tensor(name, x):
    print(
        name,
        "shape=", tuple(x.shape),
        "dtype=", x.dtype,
        "device=", x.device,
        "finite=", torch.isfinite(x).all().item(),
    )
    if x.numel() > 0 and x.is_floating_point():
        print("min/max/mean=", x.min().item(), x.max().item(), x.mean().item())
```

AI Debug 中要同时看 shape、dtype、device 和数值范围。见 [[Debug与工程基础]]。

## 4. Tensor 的索引与 shape 操作

### 4.1 索引和切片

```python
x = torch.arange(12).reshape(3, 4)

print(x[0])
print(x[:, 1])
print(x[:2, :])
```

PyTorch Tensor 的索引方式和 NumPy 很接近：

- x[0] 取第一行。
- x[:, 1] 取所有行的第二列。
- x[:2, :] 取前两行。

### 4.2 reshape、view 和 flatten

```python
x = torch.arange(24)
y = x.reshape(2, 3, 4)
z = y.flatten(start_dim=1)

print(y.shape)
print(z.shape)
```

flatten 从指定维度开始合并。图像模型中常见把 B,C,H,W 变成 B, C×H×W。

view 也能改变形状，但要求内存布局满足条件。初学时优先理解 reshape 的语义；遇到 contiguous/view 错误时再进一步检查内存布局。

### 4.3 unsqueeze 与 squeeze

```python
x = torch.zeros(224, 224, 3)
batched = x.unsqueeze(0)
print(batched.shape)

restored = batched.squeeze(0)
print(restored.shape)
```

unsqueeze 添加长度为 1 的维度；squeeze 删除长度为 1 的维度。注意 batch size 恰好为 1 时，盲目使用 squeeze 可能把真正的 batch 维删掉。

### 4.4 permute

```python
image_hwc = torch.zeros(224, 224, 3)
image_chw = image_hwc.permute(2, 0, 1)
print(image_chw.shape)
```

permute 调整维度顺序。NumPy 中常见 transpose；PyTorch 中常见 permute。HWC 与 CHW 的意义见 [[NumPy]]。

## 5. 运算、广播与矩阵乘法

### 5.1 逐元素运算

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([10.0, 20.0, 30.0])

print(x + y)
print(x * y)
print(x ** 2)
```

逐元素运算要求 shape 兼容。广播规则与 NumPy 相近。

```python
x = torch.ones(32, 128)
bias = torch.zeros(128)
y = x + bias
print(y.shape)
```

### 5.2 矩阵乘法

```python
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = a @ b
print(c.shape)
```

a 的最后一维 3 必须等于 b 的倒数第二维 3，输出 shape 是 (2,4)。

torch.matmul、@ 和 torch.mm 有不同的适用范围；当前重点是读懂矩阵维度关系，不要把 * 当作矩阵乘法。

### 5.3 reduction 与维度保留

```python
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print(x.mean())
print(x.mean(dim=0))
print(x.mean(dim=1))
print(x.mean(dim=1, keepdim=True).shape)
```

dim 表示沿哪个维度聚合；keepdim=True 保留长度为 1 的维度，便于后续广播。NumPy 的 axis 有类似作用，见 [[NumPy]]。

## 6. autograd：自动微分

### 6.1 为什么需要梯度？

模型有参数，训练的目标是调整参数，使 loss 变小。梯度告诉我们：参数改变一点时，loss 大致朝哪个方向变化。

手写每个参数的导数很困难。PyTorch autograd 会记录 Tensor 运算，并根据计算图计算梯度。

### 6.2 requires_grad

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()

print(y)
print(x.grad)
```

这里 y = x² + 3x，导数是 2x + 3；x=2 时梯度是 7。backward 之后，梯度保存在 x.grad 中。

requires_grad=True 表示需要追踪这个 Tensor 对参数的影响。模型参数默认通常需要梯度。

### 6.3 计算图

```text
x
    ↓ 运算
y = f(x)
    ↓
loss
    ↓ backward
∂loss/∂x
```

计算图是由运算连接起来的关系。只要相关 Tensor 需要梯度，PyTorch 就可以沿图反向计算。

如果调用 detach 或在 no_grad 环境中计算，就不会为该路径建立正常的梯度追踪。

### 6.4 梯度会累积

```python
x = torch.tensor(2.0, requires_grad=True)

for _ in range(2):
    y = x ** 2
    y.backward()
    print(x.grad)
```

默认情况下，梯度会累积，不会自动清零。在训练循环中必须在合适位置调用 optimizer.zero_grad，否则本次梯度会叠加之前的梯度。

## 7. nn.Module：组织模型

### 7.1 模型是什么？

模型可以理解为一个从输入到输出的函数，但它通常包含可学习参数。

PyTorch 使用 nn.Module 组织模型：

```python
import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
```

拆解：

- class SimpleModel 定义模型类。
- nn.Module 是 PyTorch 模型的基类。
- super().__init__() 初始化父类。
- self.linear 是子模块。
- forward 定义输入如何经过模型得到输出。

### 7.2 创建和调用模型

```python
model = SimpleModel(input_dim=128, output_dim=10)
x = torch.randn(32, 128)
outputs = model(x)

print(outputs.shape)
```

调用 model(x) 通常会触发模型的 forward。不要直接手动调用 model.forward(x) 代替 model(x)，因为 Module 的调用过程可能还负责 hooks 等框架行为。

### 7.3 参数

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)
```

参数是模型中会被优化器更新的 Tensor。nn.Module 会自动注册通过 self.xxx 保存的子模块和参数。

如果把层放在普通 Python list 中，PyTorch 可能无法正确注册它们；需要使用 nn.ModuleList 或 nn.Sequential 等模块容器。

### 7.4 常见层

```python
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

Sequential 按顺序连接模块。真实项目可能使用更复杂的 class、残差连接、卷积、注意力或多分支结构；阅读时仍先追踪每个模块的输入 shape 和输出 shape。

## 8. Dataset 与 DataLoader

### 8.1 Dataset 的职责

Dataset 表示“如何根据索引取得一个样本”。

```python
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, n=100):
        self.x = torch.randn(n, 128)
        self.y = torch.randint(0, 10, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

三个核心部分：

- __init__：准备数据和配置。
- __len__：数据集有多少样本。
- __getitem__：根据索引返回一个样本。

Dataset 不等于 batch；它通常提供单个样本，DataLoader 再把样本组成 batch。

### 8.2 检查一个样本

```python
dataset = ToyDataset(n=100)
sample = dataset[0]

print(type(sample))
print(sample[0].shape)
print(sample[1].shape)
```

先检查 dataset[0]，比直接启动完整训练更容易定位数据问题。

### 8.3 DataLoader 的职责

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

batch_x, batch_y = next(iter(loader))
print(batch_x.shape)
print(batch_y.shape)
```

DataLoader 负责：

- 按 batch 取样本。
- 可选随机打乱。
- 组织多个 worker。
- 使用 collate 规则组合样本。

对 batch 运行前要检查：第一维是否是 batch，标签 shape 是否匹配，dtype 是否适合 loss。

## 9. loss、optimizer 与训练循环

### 9.1 loss 是什么？

**loss / 损失** 是衡量预测与目标差异的数值。训练通常希望最小化 loss。

分类示例：

```python
model = nn.Linear(128, 10)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(32, 128)
target = torch.randint(0, 10, (32,))
logits = model(x)
loss = loss_fn(logits, target)

print(logits.shape)
print(loss.shape)
print(loss.item())
```

常见语义：

- logits shape 是 B,C。
- target shape 是 B。
- C 是类别数。
- target 通常是类别索引，而不是 one-hot，具体以 loss 文档和项目代码为准。
- loss 通常是标量 Tensor。

### 9.2 optimizer 是什么？

optimizer 根据参数的梯度更新参数。

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)
```

model.parameters() 提供需要更新的参数；lr 是学习率。学习率太大可能导致不稳定，太小可能学习缓慢。

### 9.3 一次训练更新

```python
model.train()

optimizer.zero_grad()
logits = model(x)
loss = loss_fn(logits, target)
loss.backward()
optimizer.step()
```

顺序非常重要：

1. model.train() 设置训练模式。
2. zero_grad 清理上一轮梯度。
3. forward 计算输出。
4. loss_fn 计算损失。
5. backward 计算梯度。
6. step 根据梯度更新参数。

如果把 step 放在 backward 前，优化器没有新梯度可用；如果忘记 zero_grad，梯度可能累积。

### 9.4 完整最小训练循环

```python
for epoch in range(5):
    model.train()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)

        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    print(f"epoch={epoch}, loss={loss.item():.4f}")
```

这只是结构示意。真实项目还会记录平均 loss、评估指标、学习率、checkpoint 和随机种子。

## 10. train、eval 与 no_grad

### 10.1 model.train()

model.train() 告诉模块进入训练模式。某些层在训练与评估时行为不同，例如 Dropout 和 BatchNorm。

### 10.2 model.eval()

model.eval() 切换到评估模式：

```python
model.eval()
with torch.no_grad():
    logits = model(batch_x)
```

### 10.3 torch.no_grad()

torch.no_grad() 暂停梯度记录，常用于 evaluation 和推理，可以降低内存使用。

train/eval 和 no_grad 不是一回事：

| 操作 | 作用 |
|---|---|
| model.train() | 切换模块行为到训练模式 |
| model.eval() | 切换模块行为到评估模式 |
| torch.no_grad() | 不记录梯度计算图 |

评估通常需要两者：

```python
model.eval()
with torch.no_grad():
    predictions = model(batch_x)
```

## 11. device 与 GPU

### 11.1 选择设备

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

模型和输入需要一致：

```python
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)
outputs = model(batch_x)
```

不要只把模型移到 GPU，却把输入留在 CPU。

### 11.2 验证 GPU

```python
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

False 可能表示没有 GPU、驱动不可用、PyTorch 构建不匹配或服务器没有分配 GPU，不一定是 Python 代码错误。环境层排查见 [[Conda与环境管理]]。

### 11.3 CPU 与 GPU 结果

CPU/GPU 计算可能存在微小数值差异，尤其在并行和混合精度场景。实验记录中应保存设备、PyTorch 版本、随机种子和配置。

## 12. checkpoint：保存可以恢复的状态

### 12.1 什么是 checkpoint？

checkpoint / 检查点是训练过程中保存的状态，常包含：

- 模型参数。
- 优化器状态。
- 当前 epoch 或 step。
- 最佳指标。
- 配置或随机状态。

它不是完整项目快照，也不自动包含源代码、环境、数据和 GPU 驱动。

### 12.2 保存和加载模型参数

```python
torch.save(model.state_dict(), "model.pt")

model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
```

state_dict 可以理解为“参数名到参数 Tensor 的字典”。保存参数通常比直接保存整个 Python 对象更容易迁移，但模型结构代码仍必须存在并兼容。

### 12.3 保存完整训练状态

```python
checkpoint = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "loss": loss.item(),
}

torch.save(checkpoint, "checkpoint.pt")
```

加载时：

```python
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1
```

同时保存 Git commit、配置、环境版本、数据版本和启动命令，才能更接近可复现。

## 13. AI Research Code 的完整视角

一个典型训练入口可以抽象为：

```python
def main(config):
    dataset = build_dataset(config)
    loader = DataLoader(dataset, batch_size=config["batch_size"])
    model = build_model(config).to(config["device"])
    optimizer = build_optimizer(model, config)

    for epoch in range(config["epochs"]):
        train_one_epoch(model, loader, optimizer, config["device"])
        evaluate(model, loader, config["device"])
        save_checkpoint(model, optimizer, epoch, config)
```

阅读顺序：

1. config 定义了哪些参数？
2. Dataset 返回什么样本？
3. DataLoader 组成什么 batch？
4. model 接收什么 shape，返回什么 shape？
5. loss 如何使用 output 和 target？
6. optimizer 更新哪些参数？
7. train/eval 模式在哪里切换？
8. checkpoint 保存了什么？
9. 数据、代码、环境和结果如何对应？

### 13.1 视觉数据

常见输入：

```text
单张图像：C,H,W 或 H,W,C
图像 batch：B,C,H,W 或 B,H,W,C
```

PyTorch 模型常见 BCHW，但项目可能在 Dataset 或 transform 中完成转换。不要在训练循环中随意 permute，先找到数据接口的约定。

### 13.2 轨迹与动作数据

具身智能中可能看到：

```text
状态：B,T,D_state
动作：B,T,D_action
观测：B,T,C,H,W
```

B 是 batch，T 是时间，D 是特征维度。World Model、imitation learning 和 latent planning 经常需要同时对齐 batch 和时间维度。

## 14. 常见错误与 Debug

### 14.1 shape mismatch

- **现象**：linear、matmul 或卷积层提示维度不匹配。
- **原因**：输入 shape、轴顺序、模型配置或 checkpoint 不一致。
- **判断**：打印每个关键模块的输入输出 shape，确认 B/C/H/W 或 B/T/D 语义。
- **处理**：沿数据流向前追踪；使用匹配的配置或明确的 reshape/permute，不要只修改最后一层数字。
- **为什么**：报错的位置可能只是第一个发现不一致的位置。

### 14.2 device mismatch

- **现象**：提示 Tensor 不在同一 device。
- **原因**：模型、输入、标签、hidden state 或新创建 Tensor 分布在不同设备。
- **判断**：打印 model 参数 device、输入 device 和中间变量 device。
- **处理**：集中管理 device，把参与计算的对象移动到兼容设备；不要在代码各处随意 .cuda()。

### 14.3 dtype mismatch

- **现象**：操作或 loss 提示 dtype 不符合要求。
- **原因**：输入是整数/半精度，模型或损失期望浮点；分类标签类型不符合接口。
- **判断**：打印输入、输出、参数和 target dtype。
- **处理**：按照操作和 loss 的要求显式转换，并确认转换没有改变语义。

### 14.4 loss 为 NaN

- **现象**：第一步或训练中 loss 变成 nan。
- **原因**：输入异常、非法数学操作、学习率过大、梯度爆炸或混合精度问题。
- **判断**：依次检查输入、输出、loss 输入和 loss 的 torch.isfinite。
- **处理**：定位第一次出现非有限值的操作，再处理数据、模型或优化参数；不要直接把 NaN 替换为 0。

### 14.5 CUDA out of memory

- **现象**：显存不足。
- **原因**：batch、分辨率、序列长度、模型或保存计算图造成显存超过上限。
- **判断**：用 batch size 1 做定位；查看输入 shape 和设备；检查 evaluation 是否使用 no_grad。
- **处理**：减少 batch 或输入规模、使用梯度累积/混合精度等，并记录对训练动态的影响。
- **为什么**：降低 batch 可能让代码运行，但不一定保持原实验设置。

### 14.6 loss 不下降

- **现象**：代码能跑，但 loss 不降或指标接近随机。
- **原因**：标签错位、数据归一化、模型输出和 loss 不匹配、学习率、梯度或 train/eval 模式错误。
- **判断**：先尝试让模型过拟合极小数据集；检查梯度是否存在、参数是否变化；人工查看样本和标签。
- **处理**：按数据 → forward → loss → backward → optimizer 的顺序缩小范围。
- **为什么**：能完成一次训练循环不代表科学逻辑正确。

### 14.7 checkpoint 加载失败

- **现象**：missing keys、unexpected keys 或 size mismatch。
- **原因**：模型结构、参数命名、配置、代码 commit 与保存 checkpoint 时不一致。
- **判断**：查看 checkpoint 键、当前 model.state_dict 键、保存时的配置和 commit。
- **处理**：使用匹配的模型定义；只有在理解差异后才使用 strict=False 等选项。
- **为什么**：忽略缺失参数可能让模型处于未正确初始化状态。

## 15. 一个完整的最小 Debug 流程

当训练代码复杂时，不要直接运行完整实验。按层验证：

```text
1. import torch 是否成功
2. 创建一个 Tensor
3. Dataset[0] 是否成功
4. next(iter(loader)) 是否成功
5. 打印 batch shape/dtype/device
6. model(batch) 是否成功
7. loss 是否为有限标量
8. backward 是否成功
9. 参数是否真的变化
10. eval 和 checkpoint 是否成功
```

参数是否变化可以简单检查：

```python
before = next(model.parameters()).detach().clone()

optimizer.zero_grad()
outputs = model(batch_x)
loss = loss_fn(outputs, batch_y)
loss.backward()
optimizer.step()

after = next(model.parameters()).detach()
print(torch.equal(before, after))
```

如果参数始终不变，要检查 requires_grad、optimizer 是否拿到了正确参数、loss 是否连接到模型输出以及 step 是否执行。

## 16. 与其他笔记的关系

本篇重点负责：

- Tensor、shape、dtype、device、batch
- autograd 和计算图
- nn.Module、参数和 forward
- Dataset、DataLoader
- loss、optimizer 和训练循环
- train/eval、no_grad、GPU
- checkpoint 与 AI 训练 Debug

不在本篇系统展开：

- Python 语法、函数、类和模块 → [[Python基础]]
- NumPy 数组、广播和基础预处理 → [[NumPy]]
- Conda、pip、解释器和 CUDA 环境安装 → [[Conda与环境管理]]
- Shell、路径、后台进程 → [[Linux与Shell]]
- traceback、日志和通用排错方法 → [[Debug与工程基础]]
- Git commit、分支和实验版本 → [[Git]]

本篇会出现视觉、轨迹、World Model 和 latent planning 的数据 shape，但不在这里展开模型架构、论文理论或机器人控制算法。

## 17. 本篇知识地图

```text
PyTorch 训练
├── 数据
│   ├── Dataset
│   ├── DataLoader
│   └── batch Tensor
├── Tensor 基础
│   ├── shape
│   ├── dtype
│   ├── device
│   └── 广播/维度操作
├── 模型
│   ├── nn.Module
│   ├── parameters
│   └── forward
├── 学习
│   ├── outputs
│   ├── loss
│   ├── autograd/backward
│   └── optimizer/step
├── 模式
│   ├── train
│   ├── eval
│   └── no_grad
└── 工程
    ├── GPU
    ├── checkpoint
    ├── config
    └── Debug/复现
```

## 18. 快速查询表

| 我要做什么 | 方法 | 作用 |
|---|---|---|
| 创建随机 Tensor | torch.randn(shape) | 创建浮点随机数据 |
| 查看状态 | shape、dtype、device | 检查 Tensor 接口 |
| 移动设备 | x.to(device) | 移到 CPU/GPU |
| 增加维度 | x.unsqueeze(dim) | 添加长度为 1 的轴 |
| 删除维度 | x.squeeze(dim) | 删除长度为 1 的轴 |
| 调整顺序 | x.permute(...) | 改变维度顺序 |
| 计算梯度 | loss.backward() | 反向传播 |
| 清理梯度 | optimizer.zero_grad() | 避免梯度累积 |
| 更新参数 | optimizer.step() | 使用梯度更新模型 |
| 训练模式 | model.train() | 启用训练行为 |
| 评估模式 | model.eval() | 启用评估行为 |
| 关闭梯度 | with torch.no_grad() | 减少评估开销 |
| 保存参数 | torch.save(model.state_dict(), path) | 保存模型参数 |
| 检查有限值 | torch.isfinite(x).all() | 检查 NaN/Inf |

## 19. 练习题（题目下附参考答案）

### Level 1：概念

1. Tensor 是什么？它与 NumPy array 有什么关系？

   **参考答案：**Tensor 是 PyTorch 的多维数值对象，除了 shape 和 dtype，还能携带 device 和自动微分信息；NumPy array 是 CPU 数值数组，二者可以转换但用途和计算图能力不同。

2. shape、dtype、device 分别描述什么？

   **参考答案：**shape 描述各维长度；dtype 描述元素类型；device 描述数据位于 CPU 还是 GPU。

3. batch size 是什么？为什么会影响显存？

   **参考答案：**batch size 是一次并行处理的样本数；样本越多，中间激活、梯度和临时 Tensor 通常越多，因此显存使用会增加。

### Level 2：判断

4. 判断：Tensor 的第一维一定是 batch。
   
   **参考答案：**错误。第一维是否是 batch 要看项目约定；它可能是通道、时间或其他维度。

5. 判断：调用 loss.backward() 后，optimizer 会自动更新参数。
   
   **参考答案：**错误。backward 计算梯度；还需要 optimizer.step() 才会根据梯度更新参数。

6. 判断：model.eval() 等价于 torch.no_grad()。
   
   **参考答案：**错误。eval 切换模块行为；no_grad 关闭梯度记录。评估通常需要二者同时使用。

7. 判断：checkpoint 文件包含了全部代码、环境和数据。
   
   **参考答案：**错误。checkpoint 通常保存模型/优化器状态，代码、配置、环境和数据仍需单独记录。

### Level 3：读代码

8. 解释 x = torch.randn(32, 128) 中两个数字的常见含义。
   
   **参考答案：**常见含义是 32 个样本组成一个 batch，每个样本有 128 个特征；具体语义要看项目。

9. 下面代码为什么可能报 device 错误？
   
   ```python
   model = model.to("cuda")
   outputs = model(x)
   ```
   
   **参考答案：**模型在 GPU，但 x 可能仍在 CPU。应检查并把输入移动到相同 device，例如 x = x.to(device)。

10. 解释一次参数更新的顺序。
    
    ```python
    optimizer.zero_grad()
    outputs = model(x)
    loss = loss_fn(outputs, target)
    loss.backward()
    optimizer.step()
    ```
    
    **参考答案：**先清理旧梯度，再 forward 得到输出，计算 loss，反向传播计算梯度，最后 optimizer 根据梯度更新参数。

11. Dataset 和 DataLoader 的职责有什么不同？
    
    **参考答案：**Dataset 定义如何根据索引取得一个样本以及数据集长度；DataLoader 把样本组织成 batch，并处理打乱、迭代和部分加载逻辑。

### Level 4：应用

12. 为什么先检查 dataset[0] 再启动训练？
    
    **参考答案：**可以先验证样本类型、字段、shape、dtype、标签和路径，缩小问题范围，避免完整训练跑很久才发现数据接口错误。

13. 一个分类模型输出 shape 为 B,C，标签 shape 通常是什么？
    
    **参考答案：**使用 CrossEntropyLoss 等常见分类损失时，标签通常是 B 个类别索引，具体 dtype 和语义要遵守所用 loss 的接口。

14. 如何写一个最小的 nn.Module？
    
    **参考答案：**继承 nn.Module，在 __init__ 中调用 super 并定义层，在 forward 中定义输入如何经过这些层得到输出，然后实例化并调用 model(x)。

### Level 5：Debug

15. loss 是 NaN，给出合理排查顺序。
    
    **参考答案：**依次检查输入、模型输出、loss 输入和 loss 的 isfinite；同时检查数值范围、学习率、初始化、非法数学运算和混合精度，定位第一次出现非有限值的位置。

16. 显存不足时，降低 batch size 能证明什么，不能证明什么？
    
    **参考答案：**它可以验证显存压力是否是主要因素；不能证明模型逻辑、数据语义和最终实验设置正确，因为 batch size 变化可能改变训练动态。

17. checkpoint 加载出现 size mismatch，应该先看什么？
    
    **参考答案：**比较保存时和当前模型的结构、配置、参数 shape、代码 commit 和 checkpoint 键名；不要先使用 strict=False 隐藏差异。

### Level 6：综合科研场景

18. 一个具身智能项目的观测 shape 为 B,T,C,H,W，动作 shape 为 B,T,D。你会检查哪些维度关系？
    
    **参考答案：**检查 batch B 和时间 T 是否对齐，图像 C/H/W 是否符合模型约定，动作 D 是否对应动作空间；同时检查 dtype、device、时间窗口和 mask。

19. 设计从 NumPy 图像到 PyTorch 模型的一组检查步骤。
    
    **参考答案：**检查 NumPy shape、dtype、范围和有限值；按项目规范从 HWC/BHWC 转为 CHW/BCHW；转换为 float Tensor；移动到 device；再次检查 Tensor 状态；用一个 batch 做最小 forward。

20. 设计一次“单 batch 训练验证”，证明模型、loss 和 optimizer 基本连通。
    
    **参考答案：**取一个 batch → 检查 shape/dtype/device → model.train → zero_grad → forward → 计算标量且有限的 loss → backward → 检查梯度 → step → 比较参数前后是否变化；再用 eval/no_grad 做一次推理。

## 20. 掌握度检查（附答案）

1. Tensor 与 NumPy array 的核心区别是什么？

   **答案：**Tensor 支持 PyTorch 的计算图、GPU 和模型训练流程；NumPy array 主要是 CPU 数值数组。二者可以转换，但转换可能涉及 dtype、设备和共享内存问题。

2. 如何解释一个 Tensor 的 shape？

   **答案：**不仅要列出数字，还要为每一维赋予语义，例如 B、C、H、W 或 B、T、D，并确认它符合下一个模块的接口。

3. dtype 为什么会导致 PyTorch 报错？

   **答案：**不同运算、模型层和 loss 对输入类型有要求；整数、浮点、半精度和标签索引不能随意互换。

4. device mismatch 是什么？

   **答案：**参与同一计算的模型、输入或中间 Tensor 位于不同设备，例如模型在 GPU、输入在 CPU。

5. autograd 解决什么问题？

   **答案：**它记录需要追踪的运算并在 backward 时计算 loss 对模型参数的梯度，减少手工求导工作。

6. 为什么训练循环要 zero_grad？

   **答案：**PyTorch 默认累积梯度；zero_grad 清理上一轮梯度，避免本轮更新混入不需要的历史梯度。

7. forward、loss、backward、step 分别做什么？

   **答案：**forward 计算预测；loss 衡量预测与目标差异；backward 计算梯度；step 使用梯度更新参数。

8. nn.Module 的作用是什么？

   **答案：**组织模型层、参数和 forward 计算，使 PyTorch 能够注册参数、移动设备、切换 train/eval 并配合优化器。

9. Dataset 和 DataLoader 如何连接？

   **答案：**Dataset 提供单个样本；DataLoader 按索引迭代 Dataset，把样本组成 batch 并提供训练循环需要的迭代接口。

10. model.train()、model.eval()、torch.no_grad() 有什么区别？

    **答案：**train/eval 切换层的训练或评估行为；no_grad 关闭梯度记录。评估通常使用 eval 加 no_grad。

11. 为什么 Dataset[0] 是重要的 Debug 起点？

    **答案：**它能在训练前验证路径、样本结构、字段、shape、dtype 和标签，快速缩小数据问题范围。

12. CUDA OOM 的第一轮排查应包括什么？

    **答案：**确认是显存不足，检查 batch、分辨率、序列长度、模型和是否错误保存计算图；用小 batch 做定位，并记录设置变化。

13. checkpoint 为什么不能单独保证实验复现？

    **答案：**它通常不包含完整代码、环境、配置、数据、随机种子和硬件信息；这些必须与 checkpoint 一起记录。

14. 如何检查一次 optimizer.step() 是否真的更新了参数？

    **答案：**在 step 前复制某个参数，完成 backward 和 step 后比较参数；同时检查 requires_grad、梯度是否存在以及 optimizer 是否持有正确参数。

15. 从原始数据到训练结果，PyTorch 的完整链路是什么？

    **答案：**数据读取与预处理 → Dataset → DataLoader → batch Tensor → model forward → loss → backward/autograd → optimizer.step → checkpoint/evaluation，并在每层验证 shape、dtype、device 和数值。
