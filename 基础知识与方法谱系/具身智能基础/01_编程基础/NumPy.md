# NumPy：AI 代码中的数组与数据基础

NumPy 是 Python 科研计算中最常见的基础库之一。它提供 ndarray（多维数组）、数组运算、索引、切片、广播、统计和线性代数能力。许多视觉、机器人和深度学习代码都会先用 NumPy 读取或处理数据，再把数组转换成 PyTorch Tensor。

如果不理解 NumPy，后续看到 shape、batch、channel、dtype、transpose、reshape 或数据归一化时，很容易只记住命令，却不知道每一维代表什么。

> [!NOTE]
> 本篇重点负责 NumPy 数组、shape、dtype、索引、切片、广播、拼接、统计、随机数和与 PyTorch 的边界。Python 语法见 [[Python基础]]；命令行和数据路径见 [[Linux与Shell]]；通用 Debug 见 [[Debug与工程基础]]；Tensor、autograd、Dataset/DataLoader 和训练见 [[PyTorch]]。

## 0. 学习目标

学完以后，应该能够：

- 解释 ndarray、维度、轴、shape、dtype 和元素。
- 创建数组并查看 shape、dtype、ndim、size。
- 读取、修改、切片和遍历数组。
- 理解视图与副本的基本差异。
- 使用 reshape、transpose、expand_dims 和 squeeze 处理数据布局。
- 理解广播以及为什么数组相加有时会失败。
- 使用 concatenate、stack、where 和基本统计。
- 处理图像、轨迹、状态和 batch 数据的常见形状。
- 判断什么时候应该保持 NumPy，什么时候转换为 PyTorch Tensor。
- 在 AI/具身智能代码中检查 shape、dtype、数值范围和 NaN。

## 1. 为什么需要数组？

假设一个 RGB 图像有高度 224、宽度 224、3 个颜色通道。用 Python 列表存储它可以做到，但列表嵌套很慢，难以表达统一的数学运算：

```text
每个像素乘以 1/255
所有像素减去均值
一批图像统一计算统计量
两个矩阵相乘
```

NumPy 提供一种更适合数值计算的结构：

```python
import numpy as np

image = np.zeros((224, 224, 3), dtype=np.uint8)
normalized = image.astype(np.float32) / 255.0
```

这里 image 是一个三维数组，不是三张互不相关的列表。NumPy 可以对整个数组执行向量化运算，减少手写循环。

在 AI 项目中，NumPy 常见于：

- 读取图像、深度图、传感器记录和轨迹。
- 预处理数据并保存为 npy/npz。
- 组织 batch、时间序列和状态向量。
- 在送入模型前检查 shape、dtype 和数值范围。
- 将数据转换成 PyTorch Tensor。

## 2. 核心概念

### 2.1 ndarray（多维数组）

NumPy 的核心对象是 ndarray，意思是 N-dimensional array，即 N 维数组。

直觉：它是一个规则的“网格”，每个位置存放一个值。

```python
import numpy as np

x = np.array([1, 2, 3])
print(x)
```

x 有 3 个元素，是一维数组。二维数组可以理解为表格：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

它有 2 行、3 列。

### 2.2 元素、维度、轴与 shape

**元素 / element** 是数组中的一个值。

**维度 / dimension** 描述数组需要几个索引才能定位一个元素。

**轴 / axis** 是数组的某个方向。对于二维数组，axis 0 通常表示行方向，axis 1 通常表示列方向。

**shape** 是每个轴的长度。

```python
x = np.zeros((2, 3, 4))
print(x.ndim)   # 3
print(x.shape)  # (2, 3, 4)
print(x.size)   # 24
```

这个 shape 表示：

- axis 0 长度是 2。
- axis 1 长度是 3。
- axis 2 长度是 4。
- 总元素数是 2 × 3 × 4 = 24。

shape 不是“数组里有多少个数字”这么简单；它还表达数据结构和语义。

### 2.3 用 AI 数据理解 shape

| 数据 | 常见 shape | 语义 |
|---|---|---|
| 灰度图 | H, W | 高、宽 |
| RGB 图像 | H, W, C | 高、宽、通道 |
| 一批图像 | B, H, W, C | batch、高、宽、通道 |
| 状态向量 | D | D 个特征 |
| 一批状态 | B, D | batch、特征 |
| 轨迹 | T, D | 时间步、特征 |
| 一批轨迹 | B, T, D | batch、时间步、特征 |

B 通常是 batch size，T 是时间长度，D 是特征维度，H/W 是图像高度和宽度，C 是通道数。具体项目可能采用不同布局，必须看代码和数据声明，不能只凭习惯猜。

### 2.4 dtype（数据类型）

dtype 表示数组元素的类型。

常见类型：

| dtype | 用途 |
|---|---|
| int32/int64 | 整数、索引、类别标签 |
| float32 | 常见模型输入和连续数值 |
| float64 | 更高精度的科学计算 |
| uint8 | 常见原始图像像素 0 到 255 |
| bool | 条件掩码 |

```python
image = np.zeros((2, 2, 3), dtype=np.uint8)
state = np.zeros(10, dtype=np.float32)

print(image.dtype)
print(state.dtype)
```

dtype 会影响内存、运算结果和与 PyTorch 的兼容方式。原始图像是 uint8，不代表模型应该直接使用 uint8；通常需要转为浮点并按项目要求归一化。

## 3. 创建数组

### 3.1 从 Python 数据创建

```python
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])
```

嵌套列表应当形状规则。混合长度可能产生 object 数组或警告，不适合作为模型输入。

### 3.2 创建全零、全一和固定值数组

```python
zeros = np.zeros((2, 3), dtype=np.float32)
ones = np.ones((2, 3), dtype=np.float32)
full = np.full((2, 3), 7, dtype=np.int64)
empty = np.empty((2, 3), dtype=np.float32)
```

empty 只分配空间，不保证初始值为零。不要读取 empty 后假设它已经初始化。

### 3.3 创建序列

```python
a = np.arange(0, 10, 2)
b = np.linspace(0.0, 1.0, 5)
```

arange 按步长生成，终点通常不包含；linspace 按数量在区间内生成。浮点数循环更适合先考虑 linspace，因为 arange 的浮点步长可能出现预期外的终点。

### 3.4 随机数组

```python
rng = np.random.default_rng(42)
x = rng.normal(size=(4, 3)).astype(np.float32)
y = rng.integers(0, 10, size=(4,))
```

固定种子有助于复现实验，但不保证所有硬件、库版本和并行操作完全一致。科研记录中还需保存代码版本、环境和配置。

## 4. 查看数组状态

```python
x = np.zeros((32, 128), dtype=np.float32)

print("shape:", x.shape)
print("ndim:", x.ndim)
print("size:", x.size)
print("dtype:", x.dtype)
print("min/max:", x.min(), x.max())
print("finite:", np.isfinite(x).all())
```

在 Debug 时不要只打印数组全部内容。优先检查：

- shape 是否符合语义。
- dtype 是否符合后续运算。
- min/max/mean 是否合理。
- 是否含 NaN 或 Inf。
- 是否有空数组。
- 是否有异常大的值。

这与 [[Debug与工程基础]] 中的证据思维一致。

## 5. 索引、切片与掩码

### 5.1 一维索引

```python
x = np.array([10, 20, 30, 40])

print(x[0])    # 第一个
print(x[-1])   # 最后一个
x[1] = 99
```

Python 和 NumPy 索引从 0 开始。x[0] 是第一个元素，不是编号为 1 的元素。

### 5.2 二维索引

```python
matrix = np.arange(12).reshape(3, 4)

print(matrix[0, 1])
print(matrix[1])
print(matrix[:, 2])
```

matrix[row, column] 选择一个位置；matrix[1] 选择一整行；matrix[:, 2] 选择所有行的第 3 列。

### 5.3 切片

```python
x = np.arange(10)

print(x[2:6])
print(x[:3])
print(x[::2])
print(x[::-1])
```

切片通常是 start 包含、stop 不包含。x[2:6] 包含索引 2、3、4、5。

二维切片：

```python
image = np.zeros((224, 224, 3))
top_left = image[:112, :112, :]
red_channel = image[:, :, 0]
```

### 5.4 布尔掩码

```python
x = np.array([-2, -1, 0, 1, 2])
mask = x > 0
print(mask)
print(x[mask])
x[x < 0] = 0
```

布尔数组可以选择满足条件的元素。处理标签、有效观测、缺失值和阈值筛选时很常见。

### 5.5 花式索引

```python
x = np.array([10, 20, 30, 40])
indices = np.array([3, 0, 2])
print(x[indices])
```

它按给定索引选择元素。索引数组必须合法，否则会产生 IndexError。

## 6. 数组运算与向量化

### 6.1 逐元素运算

```python
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

print(x + y)
print(x * y)
print(x ** 2)
print(np.sqrt(x))
```

这些通常是逐元素运算，不是 Python 列表拼接。数组加法要求 shape 兼容。

### 6.2 向量化

不推荐对每个元素手写 Python 循环：

```python
result = []
for value in x:
    result.append(value * 2)
```

NumPy 常写成：

```python
result = x * 2
```

向量化通常更简洁，也可能更快。它并不意味着“永远没有循环”，而是把适合的底层数组操作交给 NumPy。

### 6.3 矩阵乘法

```python
a = np.ones((2, 3))
b = np.ones((3, 4))

c = a @ b
print(c.shape)  # (2, 4)
```

矩阵乘法要求 a 的最后一维 3 等于 b 的倒数第二维 3。这里输出 shape 是 2 × 4。

* 和 @ 不要混淆：

```python
elementwise = a * a
matrix_product = a @ b
```

## 7. 广播（Broadcasting）

广播是 NumPy 在某些 shape 不完全相同时，自动扩展较小数组以完成逐元素运算的规则。

最简单例子：

```python
x = np.ones((3, 4))
bias = np.array([1, 2, 3, 4])

y = x + bias
print(y.shape)
```

bias 的 shape 是 (4,)，NumPy 会把它理解为每一行都使用同一个长度为 4 的 bias。

另一个例子：

```python
x = np.zeros((2, 3, 4))
mean = np.zeros((1, 3, 4))

y = x - mean
```

广播不是任意复制。比较两个 shape 时，通常从最后一维向前对齐；对应维度相等，或其中一个为 1，才可能兼容。

```text
(2, 3, 4)
(   3, 4)  → 可广播
```

常见错误：

```python
x = np.zeros((32, 128))
bad = np.zeros((64,))
x + bad
```

32 和 64 在对齐后的维度上不兼容。

广播的科研风险是：有时它成功了，但语义错了。例如本来想按 batch 归一化，却因为 shape 写错而按特征广播。能运行不代表含义正确。

## 8. reshape、transpose 与维度操作

### 8.1 reshape

reshape 改变数组的 shape，通常不改变元素总数：

```python
x = np.arange(24)
y = x.reshape(2, 3, 4)
print(y.shape)
```

24 个元素可以变成 2 × 3 × 4。若新 shape 的元素总数不一致，会报错。

```python
x.reshape(5, 5)  # 错误：24 不能变成 25
```

-1 可以让 NumPy 自动推断一个维度：

```python
x.reshape(2, -1)
```

不要把 reshape 当成“改变数据语义”的万能命令。它只重新解释布局；你必须确认原 shape 和新 shape 的每一维意义。

### 8.2 transpose

图像和模型代码中经常需要改变维度顺序：

```python
image_hwc = np.zeros((224, 224, 3))
image_chw = np.transpose(image_hwc, (2, 0, 1))

print(image_hwc.shape)
print(image_chw.shape)
```

HWC 表示高、宽、通道；CHW 表示通道、高、宽。PyTorch 视觉模型经常使用 CHW 或 BCHW，但具体要看项目接口。

### 8.3 expand_dims 与 squeeze

```python
x = np.zeros((224, 224, 3))
batched = np.expand_dims(x, axis=0)
print(batched.shape)  # (1, 224, 224, 3)

removed = np.squeeze(batched, axis=0)
print(removed.shape)
```

expand_dims 添加长度为 1 的轴；squeeze 删除长度为 1 的轴。squeeze 不应被滥用，因为 batch size 恰好为 1 时可能误删你需要的 batch 维。

## 9. concatenate、stack 与 split

### 9.1 concatenate

concatenate 沿已有轴连接数组：

```python
a = np.zeros((2, 3))
b = np.ones((4, 3))

c = np.concatenate([a, b], axis=0)
print(c.shape)  # (6, 3)
```

除连接轴外，其他轴必须兼容。

### 9.2 stack

stack 创建一个新轴：

```python
a = np.zeros((3,))
b = np.ones((3,))

stacked = np.stack([a, b], axis=0)
print(stacked.shape)  # (2, 3)
```

concatenate 更像沿已有方向拼接；stack 更像把多个同形状样本组成一个新维度。构造 batch 时常见 stack。

### 9.3 split

```python
x = np.arange(12)
parts = np.split(x, 3)
print([part.shape for part in parts])
```

拆分前要确认长度能被合理切分，否则使用 array_split 或显式索引。

## 10. 统计、归一化与缺失值

### 10.1 axis 参数

```python
x = np.arange(6).reshape(2, 3)

print(x.mean())
print(x.mean(axis=0))
print(x.mean(axis=1))
```

对二维数组：

- 不写 axis：所有元素一起统计。
- axis=0：沿行方向聚合，保留列结构，结果 shape 为 (3,)。
- axis=1：沿列方向聚合，保留行结构，结果 shape 为 (2,)。

axis 是最容易混淆的参数之一。不要只背“axis 0 是行”；要看“哪个轴被压缩，哪些轴被保留”。

### 10.2 keepdims

```python
x = np.random.default_rng(0).normal(size=(32, 128))
mean = x.mean(axis=0, keepdims=True)
std = x.std(axis=0, keepdims=True)

normalized = (x - mean) / (std + 1e-6)
print(mean.shape)
print(normalized.shape)
```

keepdims=True 保留被聚合轴的长度 1，使后续广播更明确。

### 10.3 NaN、Inf 与有限值

```python
x = np.array([1.0, np.nan, np.inf])
print(np.isnan(x))
print(np.isinf(x))
print(np.isfinite(x))
```

训练前检查：

```python
assert np.isfinite(x).all(), "input contains NaN or Inf"
```

不要未经判断把 NaN 直接替换为 0。先确定它来自数据缺失、非法运算还是读取问题。Debug 方法见 [[Debug与工程基础]]。

## 11. NumPy 与 PyTorch 的边界

NumPy 数组通常在 CPU 内存中；PyTorch Tensor 可以参与自动微分并移动到 GPU。常见转换：

```python
import numpy as np
import torch

array = np.ones((2, 3), dtype=np.float32)
tensor = torch.from_numpy(array)

back = tensor.numpy()
```

注意：

- from_numpy 常与原 NumPy 数组共享内存；修改一方可能影响另一方。
- NumPy 数组通常不能直接参与 GPU 运算。
- dtype 要符合模型和 loss 的要求。
- 需要梯度时通常应使用 PyTorch Tensor 的计算图，而不是先转 NumPy。
- 进入 DataLoader 前要看项目的 collate 和数据类型约定。

常见流程：

```text
磁盘/传感器数据
    ↓
NumPy 读取与预处理
    ↓
检查 shape、dtype、范围
    ↓
转换成 PyTorch Tensor
    ↓
移动到 device
    ↓
模型 forward / loss / backward
```

PyTorch 细节见 [[PyTorch]]，本篇只负责转换前后的数组基础。

## 12. 真实 AI/具身智能场景

### 12.1 图像批次

假设数据读取器给出 8 张 RGB 图像：

```python
images = np.zeros((8, 224, 224, 3), dtype=np.uint8)
print(images.shape)
print(images.dtype)

images = images.astype(np.float32) / 255.0
assert np.isfinite(images).all()
```

此时 shape 是 B,H,W,C，范围大致为 0 到 1。后续模型可能要求 B,C,H,W，需要明确转换：

```python
images_chw = np.transpose(images, (0, 3, 1, 2))
print(images_chw.shape)
```

不能只看到 shape 变成了正确形式就结束，还要确认通道和数值范围语义正确。

### 12.2 机器人状态与动作

```python
states = np.zeros((32, 20), dtype=np.float32)
actions = np.zeros((32, 7), dtype=np.float32)

assert states.shape[0] == actions.shape[0]
```

32 是 batch size，20 和 7 是不同的特征维度。它们不能因为“都是二维数组”就直接相加；要先确认代码语义。

### 12.3 时间序列与轨迹

```python
trajectory = np.zeros((100, 20), dtype=np.float32)
window = trajectory[10:26]
print(window.shape)  # (16, 20)
```

如果构造 8 个长度为 16 的窗口：

```python
windows = np.stack([
    trajectory[i:i+16]
    for i in range(8)
])
print(windows.shape)  # (8, 16, 20)
```

这对应 B,T,D。World Model、imitation learning 和 latent planning 中经常出现类似结构。

## 13. 常见错误与 Debug

### 13.1 shape 与语义不匹配

- **现象**：代码能运行，但模型指标异常，或后续层报维度错误。
- **原因**：HWC/BCHW、B/T/D 等维度顺序弄错。
- **判断**：为每一维写出语义，打印关键转换前后的 shape。
- **处理**：使用 transpose 或 reshape 前先画出数据流；不要只为了消除报错任意交换轴。
- **为什么**：shape 正确的数字形式不保证语义正确。

### 13.2 reshape 后元素数量不对

- **现象**：reshape 报错，或 reshape 成功但数据含义错乱。
- **原因**：元素总数不一致，或把不同语义的轴强行合并。
- **判断**：比较 size、原 shape 和目标 shape。
- **处理**：确认 batch、时间、特征的含义；必要时使用 transpose 后再 reshape。

### 13.3 广播意外成功

- **现象**：数组运算不报错，但结果不符合预期。
- **原因**：某个长度为 1 的轴或尾部维度触发了广播。
- **判断**：打印参与运算的 shape，手动说明每一维如何对齐。
- **处理**：显式 reshape 或 expand_dims，让意图清楚；对结果做小例子验证。

### 13.4 uint8 直接进入模型

- **现象**：输入范围过大、dtype 不匹配或训练结果异常。
- **原因**：原始图像仍是 0 到 255 的整数。
- **判断**：检查 dtype、min/max 和模型接口要求。
- **处理**：按项目规范转换为浮点、归一化并转换成 Tensor；不要假设所有模型都使用同一种归一化。

### 13.5 NumPy 与 Tensor 共享内存

- **现象**：修改 Tensor 后 NumPy 数组也变了，或反过来。
- **原因**：from_numpy 等转换可能共享底层内存。
- **判断**：构造小数组分别修改并观察，确认转换方法。
- **处理**：需要独立副本时显式 copy；需要梯度或 GPU 时使用 PyTorch 流程。

### 13.6 axis 用错

- **现象**：均值/标准差 shape 不对，归一化结果异常。
- **原因**：axis 表示被聚合的轴，记忆方向时混淆。
- **判断**：写出输入 shape、被压缩的轴和输出 shape。
- **处理**：使用 keepdims=True 并检查广播关系；用很小的数组手算验证。

## 14. 与其他笔记的关系

本篇重点负责：

- ndarray、维度、轴、shape、dtype
- 索引、切片、掩码、向量化
- 广播、reshape、transpose、拼接和统计
- 图像、状态、轨迹数据的数组表示
- NumPy 到 PyTorch 的转换边界

不在本篇系统展开：

- Python 语法和类 → [[Python基础]]
- 文件路径、命令行和数据下载 → [[Linux与Shell]]
- Debug 的证据方法 → [[Debug与工程基础]]
- Tensor、autograd、Dataset/DataLoader、模型训练和 CUDA → [[PyTorch]]
- Conda 与包安装 → [[Conda与环境管理]]

本篇中出现的 B、T、D、C、H、W 只是理解数据 shape 的必要符号；Transformer、World Model 和机器人策略模型本身不在这里展开。

## 15. 本篇知识地图

```text
NumPy 数组
├── 创建
│   ├── array
│   ├── zeros/ones
│   ├── arange/linspace
│   └── random
├── 认识数组
│   ├── shape
│   ├── ndim/size
│   └── dtype
├── 选择数据
│   ├── 索引
│   ├── 切片
│   └── 布尔掩码
├── 变换数据
│   ├── reshape
│   ├── transpose
│   ├── expand_dims/squeeze
│   └── concatenate/stack
├── 数值计算
│   ├── 向量化
│   ├── 矩阵乘法
│   ├── 广播
│   └── 统计/归一化
└── AI 数据流
    ├── 图像/状态/轨迹
    ├── 检查 shape/dtype/范围
    └── NumPy → Tensor → model
```

## 16. 快速查询表

| 我要做什么 | 方法 | 作用 |
|---|---|---|
| 创建数组 | np.array(data) | 从数据创建 ndarray |
| 全零数组 | np.zeros(shape, dtype=...) | 创建初始化为 0 的数组 |
| 查看形状 | x.shape | 查看每个轴长度 |
| 查看维度 | x.ndim | 查看轴的数量 |
| 查看元素数 | x.size | 查看总元素数量 |
| 查看类型 | x.dtype | 查看元素数据类型 |
| 选一个元素 | x[i] 或 x[i, j] | 索引 |
| 选一段 | x[start:stop] | 切片 |
| 条件筛选 | x[x > threshold] | 布尔掩码 |
| 改变形状 | x.reshape(...) | 重新组织 shape |
| 改轴顺序 | np.transpose(x, axes) | 调整维度顺序 |
| 连接已有轴 | np.concatenate(xs, axis=...) | 沿已有方向连接 |
| 创建新轴组合 | np.stack(xs, axis=...) | 增加一个维度 |
| 检查有限值 | np.isfinite(x).all() | 检查 NaN/Inf |
| 转 Tensor | torch.from_numpy(x) | NumPy 转 PyTorch |

## 17. 练习题（题目下附参考答案）

### Level 1：概念

1. ndarray、shape、ndim、size、dtype 分别是什么？

   **参考答案：**ndarray 是 NumPy 的多维数组对象；shape 是每个轴的长度；ndim 是轴的数量；size 是元素总数；dtype 是元素的数据类型。

2. shape 为 B,T,D 的数组通常可以表示什么？

   **参考答案：**通常表示一批时间序列：B 是 batch size，T 是时间步数，D 是每个时间步的特征维度。

3. 为什么原始图像常见 uint8，而模型输入常见 float32？

   **参考答案：**原始图像常用 0 到 255 的整数节省存储；模型计算和归一化通常需要浮点数，因此要按项目要求转换。

### Level 2：判断

4. 判断：shape 为 2,3,4 的数组一定表示 2 张、3 高、4 宽的图像。

   **参考答案：**错误。数字本身没有固定语义，必须看数据声明和代码约定；它也可能表示 B,T,D 或其他结构。

5. 判断：reshape 一定会改变数组中元素的排列顺序。

   **参考答案：**不一定。reshape 通常改变查看数组的形状，不改变元素总数和通常的排列关系，但必须确认布局和语义。

6. 判断：两个数组相加成功，就说明它们的语义一定正确。

   **参考答案：**错误。广播可能让不匹配的数据成功运算，但结果的业务含义仍可能错误。

7. 判断：np.empty 创建的数组已经填充为零。

   **参考答案：**错误。empty 只分配空间，不保证初始值，读取前必须显式初始化。

### Level 3：读代码

8. 下面数组的 shape、ndim 和 size 是什么？

   ```python
   x = np.zeros((2, 3, 4))
   ```

   **参考答案：**shape 是 (2,3,4)，ndim 是 3，size 是 24。

9. 解释下面代码选择了什么：

   ```python
   image = np.zeros((224, 224, 3))
   channel = image[:, :, 0]
   ```

   **参考答案：**它保留所有高度和宽度位置，只选择第一个通道，结果 shape 是 (224,224)。

10. 下面代码结果的 shape 是什么？为什么？

    ```python
    x = np.zeros((32, 128))
    bias = np.zeros((128,))
    y = x + bias
    ```

    **参考答案：**y 的 shape 是 (32,128)。bias 的最后一维 128 与 x 的最后一维一致，NumPy 将它广播到 32 行。

11. a 和 b 能否做矩阵乘法？

    ```python
    a = np.zeros((2, 3))
    b = np.zeros((4, 5))
    c = a @ b
    ```

    **参考答案：**不能。a 的最后一维是 3，b 的倒数第二维是 4，不相等，矩阵乘法不兼容。

### Level 4：应用

12. 如何把一张 HWC 的 RGB 图像转换成 CHW？

    **参考答案：**使用 np.transpose(image, (2,0,1))；但还要确认模型接口确实需要 CHW，而不是只为消除 shape 报错。

13. 如何把 8 个 shape 为 T,D 的轨迹窗口组成 B,T,D？

    **参考答案：**把 8 个同 shape 窗口放入列表，使用 np.stack(windows, axis=0)，结果第一个维度就是 batch。

14. 为什么统计时常使用 keepdims=True？

    **参考答案：**它保留被聚合轴为长度 1，使均值或标准差更容易按照预期广播回原数组。

### Level 5：Debug

15. 数组运算成功但结果异常，如何排查广播问题？

    **参考答案：**打印所有输入 shape，从最后一维向前对齐，明确每一维是相等、1 还是不兼容；用很小的数组手算结果，必要时显式 reshape。

16. 模型输入报 dtype 错误，你会检查什么？

    **参考答案：**检查 NumPy 数组的 dtype、数值范围、转换到 Tensor 后的 dtype，以及模型/损失函数期望的类型；不要只调用 astype 而不确认语义。

17. NumPy 数组转 Tensor 后，修改一方影响另一方。可能原因是什么？

    **参考答案：**from_numpy 等方式可能共享内存。需要独立数据时使用 copy；需要梯度或 GPU 时按 PyTorch 的 Tensor 流程处理。

### Level 6：综合科研场景

18. 机器人轨迹 shape 为 100,20，你要构造 8 个长度为 16 的窗口。写出思路并说明结果 shape。

    **参考答案：**对起点取 trajectory[i:i+16] 得到 8 个 (16,20) 窗口，再用 np.stack 在 axis=0 组合，结果为 (8,16,20)。

19. 图像数组 shape 为 8,224,224,3，模型要求 batch-first 的 BCHW 浮点输入。你会做哪些步骤？

    **参考答案：**先检查 dtype 和范围；转为 float32 并按项目规范归一化；用 transpose 从 BHWC 调整为 BCHW，结果 (8,3,224,224)；检查有限值，再转换为 PyTorch Tensor。

20. 设计一个 NumPy 数据进入 PyTorch 模型前的检查清单。

    **参考答案：**检查文件和样本是否存在；shape 语义；dtype；min/max/mean；NaN/Inf；batch、channel、time 顺序；转换后 Tensor shape/dtype/device；最后用一个 batch 完成最小 forward。

## 18. 掌握度检查（附答案）

1. NumPy ndarray 是什么？

   **答案：**它是 NumPy 提供的规则多维数组对象，适合批量数值运算。

2. shape 表达什么？

   **答案：**表达每个轴的长度，同时帮助理解数据的结构和语义。

3. ndim 和 size 有什么区别？

   **答案：**ndim 是轴的数量；size 是所有轴长度相乘得到的元素总数。

4. axis=0 和 axis=1 应该如何理解？

   **答案：**axis 指定被操作或聚合的轴；不要只背“行/列”，要看该轴被压缩后输出 shape 如何变化。

5. dtype 为什么重要？

   **答案：**它影响内存、运算、标签/索引语义以及与 PyTorch 模型的兼容性。

6. 索引和切片有什么区别？

   **答案：**索引通常选择具体位置或轴；切片选择一段范围，并常保留数组结构。

7. 广播解决什么问题？有什么风险？

   **答案：**广播允许兼容 shape 的数组进行逐元素运算；风险是它可能在语义错误时仍然成功，造成隐蔽的数据问题。

8. reshape 能做什么，不能保证什么？

   **答案：**reshape 改变 shape 的组织方式并保持元素总数；它不能自动保证新的每一维语义正确。

9. transpose 在图像处理中为什么常见？

   **答案：**不同库使用不同维度顺序，例如 HWC 与 CHW；transpose 可以调整轴顺序，但必须遵守模型接口约定。

10. concatenate 和 stack 的区别是什么？

    **答案：**concatenate 沿已有轴连接；stack 创建新轴，把多个同 shape 数组组合起来。

11. 如何检查数组是否含 NaN 或 Inf？

    **答案：**使用 np.isfinite(x).all()，也可以分别使用 np.isnan 和 np.isinf 定位问题类型。

12. 为什么图像常要从 uint8 转为 float32？

    **答案：**uint8 适合存储原始像素，模型计算和归一化通常需要浮点；具体范围和归一化方式要按项目规范确认。

13. B,T,D 在机器人和 World Model 数据中可能表示什么？

    **答案：**B 是 batch，T 是时间步，D 是每步的状态、动作或潜在特征维度。

14. NumPy 数组什么时候转换为 PyTorch Tensor？

    **答案：**当数据要进入 PyTorch 模型、参与 Tensor 运算、自动微分或 GPU 计算时；转换前先完成必要的数组检查和预处理。

15. 怎样判断 NumPy 预处理已经正确？

    **答案：**shape 语义正确、dtype 和范围符合接口、没有 NaN/Inf、轴顺序明确、转换后的 Tensor 状态正确，并且一个小 batch 能通过最小 forward。
