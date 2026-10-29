# Debug 与工程基础

这篇笔记学习的不是“背一组报错解决命令”，而是一种面对 AI research code 的工作方法：当程序不能运行、结果不对、训练突然崩溃时，如何先描述现象，再收集证据，判断问题属于哪一层，最后用最小改动验证修复是否真的有效。

科研代码通常同时包含 Python、Shell 命令、配置文件、数据集、模型、GPU 和外部依赖。报错的位置不一定是根因。例如，训练代码报 `FileNotFoundError`，可能不是模型写错，而是启动时所在目录不对；`CUDA out of memory` 可能不是模型“坏了”，而是 batch size 太大；`ModuleNotFoundError` 可能不是包不存在，而是运行代码的 Python 与安装包的 Python 不是同一个。

> [!NOTE]
> 本篇负责通用 Debug 思维、错误分类、日志、traceback、断点、断言、最小复现和 AI 工程排查流程。Conda、解释器路径和环境激活详见 [[Conda与环境管理]]；Python 语法与模块基础见 [[Python基础]]；Shell、路径和命令行见 [[Linux与Shell]]；版本回退见 [[Git]]；数组与 `shape` 见 [[NumPy]]；Tensor、设备和训练流程见 [[PyTorch]]。

## 0. 学习目标

学完以后，应该能够：

- 区分“程序没有启动”“运行时报错”“运行成功但结果不对”三类问题。
- 读懂 Python traceback，找到真正需要先看的代码位置。
- 用清晰的现象、复现步骤、期望结果和实际结果描述 Bug。
- 区分环境、路径、依赖、输入数据、shape、设备、配置和代码逻辑问题。
- 使用 `print`、日志、`assert`、最小复现和调试器逐层缩小问题范围。
- 在 AI 训练代码中检查 batch、shape、dtype、device、loss 和梯度。
- 不把最后一行报错信息自动当成根因，也不靠“随机重装/随机改参数”碰运气。
- 形成从仓库启动到训练、验证、checkpoint 和结果检查的工程排查流程。

## 1. 为什么需要 Debug？

理想流程是：

```text
读取配置 → 加载数据 → 创建模型 → 前向计算 → 计算 loss
    → 反向传播 → 更新参数 → 保存 checkpoint → evaluation
```

真实情况中，每一步都可能失败：

| 阶段 | 可能的问题 |
|---|---|
| 启动 | 命令错误、环境未激活、入口文件不在当前目录 |
| 导入 | 包未安装、包名写错、多个 Python 串环境 |
| 配置 | 键名拼错、路径错误、参数类型不对 |
| 数据 | 文件缺失、标签越界、样本损坏、shape 不一致 |
| 模型 | 层的输入输出维度不匹配、设备不一致 |
| 训练 | loss 为 `NaN`、梯度爆炸、显存不足、学习率不合适 |
| 结果 | 代码能跑但指标异常、数据泄漏、评估方式错误 |

Debug 的目标不是让错误暂时消失，而是回答：

1. 发生了什么？
2. 在什么条件下发生？
3. 哪条证据支持我的判断？
4. 修复后如何证明问题真的解决？

## 2. 核心概念

### 2.1 Bug、Error、Exception 与 Failure

**Bug**：程序或工程中的缺陷。它可能已经导致错误，也可能暂时没有暴露。

**Error**：错误或失败现象的统称，例如路径错误、类型错误、显存不足。

**Exception（异常）**：程序运行时发现无法按当前路径继续执行，并以结构化方式报告的问题。Python 中常见 `TypeError`、`ValueError`、`KeyError`、`FileNotFoundError` 和 `RuntimeError`。

**Failure（失败）**：一次运行没有达到预期结果。程序报异常是失败的一种，但“程序正常结束、指标却是随机水平”也是 failure。

不要把“没有红色报错”理解成“实验正确”。

### 2.2 Debug

**直觉**：像侦探一样用证据定位问题。

**正式含义**：通过观察运行状态、缩小范围、提出假设、进行受控实验，定位并修复程序或工程缺陷的过程。

Debug 不是：

- 看到报错就复制最后一行到搜索框。
- 一次改十个地方，然后不知道哪个改动有效。
- 反复重装环境而不保存错误信息。
- 把所有问题归因于 CUDA、版本或“代码太复杂”。

### 2.3 期望结果与实际结果

每次 Debug 先写清楚两件事：

```text
期望：程序应该做什么？
实际：程序具体做了什么？
```

例如：

```text
期望：读取一个 batch，得到 [32, 3, 224, 224] 的图像 Tensor
实际：读取后得到 [3, 224, 224]，模型提示维度数量不对
```

这比“模型跑不起来”更有用，因为它提供了可验证的差异。

### 2.4 日志（Log）与证据

**日志 / log** 是程序在运行过程中记录的事件、状态和数值。它可以来自 `print`，也可以来自 Python 的 `logging` 模块。

好的 Debug 证据包括：

- 完整错误信息和 traceback，而不是只截最后一行。
- 执行的完整命令。
- 当前工作目录、Python 解释器路径和代码 commit。
- 输入文件、配置文件和关键参数。
- 发生错误前最后一条日志。
- 关键变量的类型、shape、dtype、device 和数值范围。

日志的价值在于让“我觉得”变成“我观察到”。

### 2.5 Traceback（回溯）与调用栈

**Traceback** 是 Python 报错时展示的调用路径。它告诉你：哪个文件调用了哪个函数，最后在哪一行触发了异常。

**调用栈 / call stack** 可以直觉理解为“程序走过的函数路线”。如果 `train.py` 调用了 `trainer.fit()`，`fit()` 又调用了 `model(batch)`，错误可能在多层函数之后才暴露。

典型结构：

```text
Traceback (most recent call last):
  File "train.py", line 42, in <module>
    main()
  File "train.py", line 35, in main
    trainer.fit()
  File "trainer.py", line 88, in fit
    loss = self.model(batch)
TypeError: ...
```

阅读方法：

1. 从上往下看调用路径，了解程序如何走到这里。
2. 优先看最下面的异常类型和消息，它通常说明失败类别。
3. 再看最靠近业务代码的文件和行号，而不是盲目修改 Python/库内部代码。
4. 回到出错行，检查输入、前置条件和调用者传入的值。

最后一行通常是“异常被报告的位置”，不一定是“错误最初产生的位置”。

### 2.6 前置条件、后置条件与不变量

**前置条件 / precondition**：函数开始前必须成立的条件。

**后置条件 / postcondition**：函数成功返回后应该成立的条件。

**不变量 / invariant**：在循环或流程中始终应该保持的条件。

例子：

```python
def normalize_image(x):
    assert x.ndim == 4          # 前置条件：有 batch 维
    assert x.dtype.is_floating_point
    y = x / 255.0
    assert 0.0 <= y.min() and y.max() <= 1.0  # 后置条件
    return y
```

断言不是为了“让代码看起来专业”，而是尽早把错误挡在离根因更近的位置。

## 3. 错误的分层模型

面对失败，先判断它属于哪一层：

```text
第 1 层：能否启动命令？
    ↓
第 2 层：解释器和依赖是否正确？
    ↓
第 3 层：路径、配置和数据是否正确？
    ↓
第 4 层：输入输出的类型、shape、dtype、device 是否正确？
    ↓
第 5 层：模型、loss、梯度和优化逻辑是否正确？
    ↓
第 6 层：实验结果和科学结论是否可信？
```

不要在第 2 层失败时直接修改模型；也不要在第 6 层指标异常时只重装依赖。

| 层级 | 典型现象 | 第一检查对象 |
|---|---|---|
| 启动 | 命令无法识别、脚本不存在 | 当前目录、命令、Shell |
| 环境 | `ModuleNotFoundError` | `sys.executable`、pip、包版本 |
| 配置/路径 | 找不到文件、键不存在 | cwd、配置键、路径存在性 |
| 数据 | batch 内容异常 | 样本、标签、shape、dtype、范围 |
| 模型 | shape/device/type 错误 | 每层输入输出、参数设备 |
| 训练 | `NaN`、OOM、loss 不降 | loss、梯度、batch size、学习率 |
| 结果 | 指标不可信 | train/eval、数据划分、指标实现 |

## 4. 标准 Debug 循环

### Step 1：稳定复现

先确认问题能否重复出现。记录：

```text
环境：哪个 Conda 环境、哪个 Python
代码：哪个 repository、哪个 commit
命令：完整启动命令
输入：数据路径、配置、随机种子
现象：完整错误或异常结果
```

如果一次成功、一次失败，先考虑随机性、并发、数据顺序、显存状态或外部资源，而不是立即修改模型。

### Step 2：缩小范围

把完整训练拆成更小的检查：

```text
能否 import？
    ↓
能否读取一个样本？
    ↓
能否读取一个 batch？
    ↓
能否完成一次 forward？
    ↓
能否计算 loss？
    ↓
能否 backward？
    ↓
能否更新一次参数？
```

最小可运行路径通常比直接启动几小时训练更有价值。

### Step 3：提出一个可检验的假设

不要只说“可能是版本问题”。写成：

```text
假设：运行时使用了系统 Python，而不是 jepa 环境。
证据：sys.executable 指向环境外路径。
实验：用 python -m pip 和当前解释器重新检查 torch。
预期：路径一致后，import torch 成功。
```

一个假设应该对应一个主要实验，避免一次改动太多变量。

### Step 4：做最小改动

优先选择可逆、范围小的改动：

- 加一条日志，而不是重写整个训练器。
- 固定一个 batch，而不是先改模型结构。
- 把 batch size 从 64 降到 1，用于验证流程，而不是宣称问题解决。
- 修正配置路径，而不是复制大量数据到随机目录。

### Step 5：验证修复

验证至少包括：

1. 原错误是否消失。
2. 原本正常的部分是否仍正常。
3. 新结果是否符合预期。
4. 是否需要补充测试或记录。

“不报错了”只是必要条件，不是充分条件。

### Step 6：记录结论

记录最终结论：

```text
现象：...
根因：...
证据：...
修复：...
验证：...
预防：增加什么日志、断言或测试？
```

## 5. 最基础的 Debug 工具

### 5.1 `print`：快速观察

```python
print(type(batch))
print(batch.shape)
print(batch.dtype)
print(batch.device)
```

适合短脚本和第一次定位。不要只打印“到了这里”，最好打印有意义的状态。

对 batch 的更完整检查：

```python
print("images:", images.shape, images.dtype, images.device)
print("labels:", labels.shape, labels.dtype, labels.min().item(), labels.max().item())
```

### 5.2 `logging`：可持续记录

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("starting training")
logger.info("batch shape=%s", tuple(images.shape))
```

**logging** 比到处散落的 `print` 更适合科研项目，因为它可以区分级别（`DEBUG`、`INFO`、`WARNING`、`ERROR`），也可以统一输出格式和写入文件。

### 5.3 `assert`：检查必须成立的条件

```python
assert images.ndim == 4, f"expected 4D images, got {images.shape}"
assert images.shape[0] == labels.shape[0]
assert images.device == labels.device
```

如果条件不成立，程序会立即失败并显示上下文。注意：不要把带副作用的操作放进 `assert`，因为某些运行方式可能关闭断言。

### 5.4 `raise`：主动报告更清楚的错误

```python
if not config_path.exists():
    raise FileNotFoundError(
        f"config does not exist: {config_path}. "
        "Run this command from the repository root."
    )
```

相比让程序稍后在深层代码中崩溃，主动检查能让错误更接近根因。

### 5.5 调试器（Debugger）与断点

**Debugger** 是允许你暂停程序、查看变量、逐行执行和检查调用栈的工具。

最简单的 Python 断点：

```python
breakpoint()
```

程序运行到这里会暂停。你可以查看变量、执行表达式、继续运行或退出。IDE（如 VS Code）也可以用鼠标设置断点。当前阶段知道“断点让程序停在现场”即可，不必先掌握调试器全部命令。

### 5.6 Shell 检查

路径和当前工作目录是科研代码的高频问题：

```bash
pwd                         # Linux/macOS：当前目录
ls                          # Linux/macOS：列出文件
python -c "import os; print(os.getcwd())"
```

PowerShell 常见写法：

```powershell
Get-Location
Get-ChildItem
Test-Path .\configs\base.yaml
```

Shell 语法和平台差异见 [[Linux与Shell]]。

## 6. 如何阅读常见 Python 报错

### 6.1 `ModuleNotFoundError`

```text
ModuleNotFoundError: No module named 'torch'
```

先判断：

1. 当前代码使用哪个解释器？
2. 当前解释器的 pip 是否安装了该包？
3. import 名称是否和安装名称相同？例如安装名与导入名不总是一样。
4. 是否在正确的 Conda 环境中？

验证命令：

```bash
python -c "import sys; print(sys.executable)"
python -m pip show torch
python -c "import torch; print(torch.__version__)"
```

不要一看到这个错误就直接 `pip install`；先确认 pip 和 python 是否同属一个环境，详见 [[Conda与环境管理]]。

### 6.2 `FileNotFoundError`

```text
FileNotFoundError: [Errno 2] No such file or directory: 'data/train.json'
```

相对路径是相对于**当前工作目录**，不一定是相对于 Python 文件所在目录。先检查：

```python
from pathlib import Path

print("cwd:", Path.cwd())
path = Path("data/train.json")
print("resolved:", path.resolve())
print("exists:", path.exists())
```

可能的根因：没有下载数据、路径写错、从错误目录启动、配置仍是作者本机路径，或服务器挂载未完成。

### 6.3 `KeyError`

```text
KeyError: 'learning_rate'
```

这通常表示字典或配置中没有这个键。不要只在调用处加默认值；先检查配置实际包含什么：

```python
print(config.keys())
print(config)
```

还要确认键名拼写、配置文件是否被正确加载，以及不同实验配置是否使用了不同结构。

### 6.4 `TypeError`

```text
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

表示函数收到的对象类型不符合要求。检查：

```python
print(type(path), repr(path))
```

`repr` 能显示 `None`、空字符串和隐藏字符，通常比只打印值更有用。继续回溯这个值在哪里产生，往往比在报错行强行转换类型更合理。

### 6.5 `ValueError`

表示值的类型可能正确，但内容不符合要求。例如把非法字符串转成数字、输入尺寸不合法、标签范围不对。检查输入来源、允许范围和调用函数的契约。

### 6.6 `AttributeError`

```text
AttributeError: 'NoneType' object has no attribute 'to'
```

这表示你以为对象是 Tensor 或模型，但实际是 `None`。要向前追踪：哪个函数应该返回对象？是否忘写 `return`？是否某个分支没有返回？

### 6.7 `RuntimeError` 与 PyTorch 常见消息

`RuntimeError` 范围很广，不能只看异常名称。必须阅读完整消息和相关变量。

常见问题：

- `Expected all tensors to be on the same device`：Tensor 或模型分布在不同设备。
- `size mismatch`：维度或参数形状不兼容。
- `CUDA out of memory`：当前操作需要的显存超过可用显存。
- `mat1 and mat2 shapes cannot be multiplied`：矩阵乘法两侧的内部维度不匹配。

这些问题需要检查 `shape`、`dtype`、`device` 和 batch size；Tensor 细节见 [[PyTorch]]。

## 7. AI Research Code 的关键检查点

### 7.1 数据加载阶段

先只检查一个样本和一个 batch：

```python
sample = dataset[0]
print(type(sample))
print(sample)

batch = next(iter(dataloader))
print(type(batch))
```

需要确认：

- 样本数量是否正确。
- 图像、文本、动作、状态等字段是否存在。
- batch 维是否存在且大小符合预期。
- 标签是否在合法范围。
- 数据是否包含 `NaN` 或 `Inf`。
- 数据类型是否适合后续模型。

### 7.2 Shape 检查

假设视觉模型期望输入 `[B, C, H, W]`：

```python
assert images.ndim == 4
B, C, H, W = images.shape
assert C in (1, 3), f"unexpected channels: {C}"
print(B, C, H, W)
```

这里 `B` 是 batch size，`C` 是通道数，`H/W` 是高度/宽度。不要只看 Tensor 能否打印；要判断每一维的语义是否正确。shape 的系统知识见 [[NumPy]] 和 [[PyTorch]]。

### 7.3 dtype 与数值范围

```python
print(images.dtype)
print(images.min().item(), images.max().item())
assert images.dtype.is_floating_point
assert torch.isfinite(images).all()
```

例如模型可能期望浮点数，但数据仍是整数；归一化可能期望 `[0, 1]`，实际却是 `[0, 255]`。两者都可能不立即报错，却让训练结果异常。

### 7.4 device 检查

```python
print("model:", next(model.parameters()).device)
print("images:", images.device)
print("labels:", labels.device)
```

模型参数和参与同一计算的 Tensor 通常需要在兼容的设备上。不要在代码各处随意 `.cuda()`；更稳妥的是在入口处明确设备，并统一移动模型和 batch。

### 7.5 Forward、loss 与 backward 分层

把一次训练拆开：

```python
model.train()
optimizer.zero_grad(set_to_none=True)

outputs = model(images)
print("outputs:", outputs.shape, outputs.dtype, outputs.device)

loss = loss_fn(outputs, labels)
print("loss:", loss.item())
assert torch.isfinite(loss), "loss is NaN or Inf"

loss.backward()
optimizer.step()
```

如果失败：

- forward 失败：先查输入、模型结构、shape、dtype、device。
- loss 失败：查输出与标签的形状、类型和取值范围。
- backward 失败：查 loss 是否为标量、是否有非法数值、计算图是否被错误切断。
- 多次更新后异常：查学习率、梯度、数据和优化器状态。

### 7.6 `train()` 与 `eval()`

`model.train()` 和 `model.eval()` 会改变某些层的行为，例如 Dropout、BatchNorm。训练和评估阶段用错模式，代码可能不报错，但指标会不可靠。

典型评估结构：

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch)
```

检查重点：是否在 evaluation 前调用 `eval()`；是否关闭不必要的梯度；训练后是否恢复 `train()`。

### 7.7 `NaN` 与 `Inf`

**NaN** 表示“不是一个数”，**Inf** 表示无穷大。它们可能来自除零、数值溢出、非法 `log`、过大的学习率、异常输入或混合精度问题。

逐层检查：

```python
def check_finite(name, x):
    print(name, torch.isfinite(x).all().item())
    assert torch.isfinite(x).all(), f"non-finite values in {name}"

check_finite("images", images)
check_finite("outputs", outputs)
check_finite("loss", loss)
```

不要只在训练结束看 loss；要定位它第一次变成非有限值的位置。

### 7.8 显存不足（CUDA Out Of Memory）

常见现象：`CUDA out of memory`。

排查顺序：

1. 确认是显存不足，而不是设备不可用或其他 CUDA 错误。
2. 用更小 batch size 验证是否能运行。
3. 检查是否保存了不必要的计算图，例如把带梯度的 Tensor 长期放进列表。
4. 检查 evaluation 是否使用 `torch.no_grad()`。
5. 记录输入分辨率、序列长度、模型大小和显存占用。
6. 再考虑梯度累积、混合精度、梯度检查点等工程手段。

降低 batch size 是定位实验，不一定是最终科学方案；它可能改变训练动态，因此需要记录。

## 8. 工程结构与可维护性

### 8.1 入口脚本（Entry Point）

**入口脚本 / entry point** 是用户启动程序时首先执行的文件或命令，例如 `train.py`、`evaluate.py`、`scripts/train.sh`。

入口通常负责：

```text
解析参数
    ↓
读取配置
    ↓
建立环境/设备
    ↓
创建 dataset、model、optimizer
    ↓
调用训练或评估函数
```

看到入口脚本时，不要期待所有逻辑都在里面；要沿着函数调用追踪到数据、模型和训练器。

### 8.2 配置（Configuration）

配置是集中描述实验参数的文件或对象，如 YAML、JSON、命令行参数或 Python 字典。

配置 Debug 要检查：

- 实际读取的是哪个文件。
- 默认值是否覆盖了命令行值。
- 字符串数字是否被当成数字使用。
- 相对路径相对于哪里。
- 配置中的字段是否与代码版本匹配。

建议启动时打印关键配置和配置文件的绝对路径，而不是只打印“loaded config”。

### 8.3 工作目录（Working Directory）

工作目录是进程启动时的当前目录。它影响相对路径、配置查找、日志位置和输出 checkpoint 位置。

```python
from pathlib import Path
print(Path.cwd())
```

一个常见工程改进是：在入口处把项目根目录、数据根目录和输出目录显式解析成绝对路径，并在启动日志中记录。

### 8.4 版本与可回退

如果代码昨天能运行、今天不能运行，先检查：

- Git commit 是否变化。
- 依赖是否变化。
- 配置是否变化。
- 数据是否变化。
- 服务器/驱动是否变化。

不要在没有记录的情况下反复修改同一份代码。可以用 Git 保存实验性修改、查看 diff 和回退候选版本；Git 基础见 [[Git]]。

## 9. 常见错误与 Debug 过程

### 9.1 “程序什么也没输出，然后退出了”

- **现象**：命令返回成功，但没有训练日志或输出文件。
- **可能原因**：入口函数没有调用、参数分支提前返回、日志级别过高、脚本只定义了函数。
- **如何判断**：确认是否有 `if __name__ == "__main__":`；在入口第一行和关键分支加日志；检查退出码和输出目录。
- **处理**：确保入口实际调用 `main()`，确认配置没有把流程设为 dry-run 或跳过训练。
- **为什么**：Python 文件被执行不等于其中定义的函数会自动执行。

### 9.2 “代码改了但运行结果没变化”

- **现象**：修改模型或参数后输出完全相同。
- **可能原因**：运行的不是当前文件、缓存/旧安装包、参数没有传入、随机种子固定、代码分支未执行。
- **如何判断**：打印 `__file__`、配置最终值、Git diff 和关键分支日志；改变一个明显可观察的值做验证。
- **处理**：确认工作目录和 import 路径；用当前代码重新启动；清理不必要的旧安装或明确使用 editable install。
- **为什么**：研究仓库可能通过安装包、软链接或多个副本运行代码。

### 9.3 “loss 一开始就是 NaN”

- **现象**：第一个 batch 的 loss 就是 `nan`。
- **可能原因**：输入含 NaN/Inf、归一化错误、非法数学操作、标签异常或模型初始化不适合。
- **如何判断**：从数据、模型输出、loss 输入依次调用 `torch.isfinite`；检查最小 batch。
- **处理**：定位第一次出现非有限值的张量，再修正数据或数学操作；不要只把 NaN 替换成 0。
- **为什么**：替换结果可能掩盖根因并产生看似正常但不可信的实验。

### 9.4 “训练能跑，但指标非常差”

- **现象**：没有异常，loss 下降或不下降，但验证指标接近随机。
- **可能原因**：标签错位、train/eval 模式错误、数据归一化不一致、评估指标实现错误、学习率不合适或数据划分问题。
- **如何判断**：先用极小数据集过拟合；人工检查几个样本和标签；比较 train/eval 输出；检查指标输入和数据划分。
- **处理**：从最小可验证实验开始，不要先扩大模型和训练时间。
- **为什么**：能运行只说明语法和部分接口成立，不说明科学逻辑正确。

### 9.5 “同一模型在 CPU 能跑，在 GPU 不能跑”

- **现象**：切换设备后出现 CUDA 或 device 错误。
- **可能原因**：模型、输入、标签或隐藏状态不在同一设备；某个操作只支持 CPU；GPU 版本/驱动不匹配。
- **如何判断**：打印模型参数、输入、标签和中间状态的 device；确认最小 forward 是否成功。
- **处理**：统一设备管理，逐步恢复 GPU 训练；不要在每一行随意移动 Tensor。
- **为什么**：设备迁移应该是明确的工程策略，而不是零散补丁。

## 10. 一个完整的 Debug 示例

假设运行：

```bash
python train.py --config configs/base.yaml
```

出现：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x512 and 768x256)
```

正确的排查过程：

1. **描述现象**：某次 forward 在全连接层前失败；输入矩阵最后一维是 512，但层期望 768。
2. **确认环境和版本**：确认当前仓库 commit、Python 和 PyTorch，排除代码副本问题。
3. **查看模型定义**：找到 `Linear(768, 256)`，确认它确实期望 768 维输入。
4. **查看上游输出**：打印进入该层的 Tensor shape，确认 512 是真实输出而不是日志写错。
5. **向前追踪**：检查数据预处理、backbone 输出、特征拼接和 pooling；某一步可能使用了不同模型配置。
6. **检查配置**：确认 checkpoint、backbone 名称和 embedding size 与当前模型定义一致。
7. **提出假设**：预训练 checkpoint 产生 512 维特征，但当前 head 按 768 维构建。
8. **最小验证**：只加载一个 batch，打印 backbone 输出和 head 输入，验证维度关系。
9. **修复**：使用匹配的 head/checkpoint，或根据设计修改投影层；不能只把 `768` 随意改成 `512` 而不确认模型意图。
10. **回归验证**：完成一次 forward、loss、backward，再用小数据集运行若干步，并检查结果。

这个例子说明：错误消息告诉你“哪里不兼容”，但仍需要结合模型设计判断“为什么不兼容、哪一侧应该改变”。

## 11. 与其他笔记的关系

本篇重点负责：

- Debug 的证据思维和标准循环
- traceback、日志、断言、断点和最小复现
- AI 项目中的路径、配置、数据、shape、dtype、device、loss 和梯度排查
- 代码能运行但结果不可信时的工程检查

以下内容不在本篇系统展开：

- Conda、Python 解释器和 pip 串环境 → [[Conda与环境管理]]
- Python 语法、函数、模块和类 → [[Python基础]]
- Shell、路径、进程和命令行差异 → [[Linux与Shell]]
- Git commit、diff、分支和回退 → [[Git]]
- NumPy 数组与广播 → [[NumPy]]
- PyTorch Tensor、autograd、Dataset、DataLoader 和训练 API → [[PyTorch]]

本篇会使用这些概念来排查问题，但不会代替对应笔记的系统教材。

## 12. 本篇知识地图

```text
发现失败
├── 记录完整现象
│   ├── 命令
│   ├── 环境与 commit
│   ├── 输入与配置
│   └── traceback / 日志
├── 判断问题层级
│   ├── 启动
│   ├── 环境/依赖
│   ├── 路径/配置/数据
│   ├── shape/dtype/device
│   ├── 模型/loss/梯度
│   └── 结果可信度
├── 提出可检验假设
├── 最小实验缩小范围
│   ├── import
│   ├── 一个样本
│   ├── 一个 batch
│   ├── forward
│   ├── loss
│   └── backward/update
├── 最小可逆修复
└── 验证并记录
    ├── 原错误消失
    ├── 原功能仍正常
    └── 结果符合预期
```

## 13. 快速查询表

| 我要做什么 | 方法 | 作用 |
|---|---|---|
| 看完整错误 | 保留完整 traceback | 判断调用路径和异常类型 |
| 看当前 Python | `python -c "import sys; print(sys.executable)"` | 排除串环境 |
| 看当前目录 | `Path.cwd()` / `Get-Location` | 排除相对路径问题 |
| 看文件是否存在 | `Path(path).exists()` / `Test-Path` | 验证数据和配置路径 |
| 看对象类型 | `type(x)`、`repr(x)` | 排除 `None`、字符串/数字混用 |
| 看 Tensor 状态 | `shape`、`dtype`、`device` | 排除输入接口问题 |
| 看数值是否合法 | `torch.isfinite(x).all()` | 定位 NaN/Inf |
| 让错误更早出现 | `assert condition` | 检查前置条件 |
| 暂停现场 | `breakpoint()` | 查看变量和调用栈 |
| 缩小问题 | 只跑一个样本/一个 batch | 建立最小复现 |
| 查代码变化 | `git diff`、`git log` | 确认运行版本和改动 |

## 14. 练习题（题目下附参考答案）

### Level 1：概念

1. Debug 的目标是什么？为什么“程序不报错”不一定表示成功？

   **参考答案：**Debug 的目标是用证据定位并修复问题，同时验证修复没有破坏其他功能。程序不报错只说明没有触发已知异常，数据、指标和科学逻辑仍可能错误。

2. Bug、Exception 和 Failure 有什么区别？

   **参考答案：**Bug 是缺陷；Exception 是程序运行时报告问题的一种机制；Failure 是没有达到预期结果。程序可以没有 Exception 但仍然 Failure。

3. 什么是 traceback？阅读时为什么不能只看最后一行？

   **参考答案：**Traceback 是程序经过的调用路径。最后一行通常说明异常类型和消息，但调用路径和出错代码上下文有助于判断根因。

### Level 2：判断

4. 判断：报错发生在哪一行，根因就一定在那一行。

   **参考答案：**错误。那一行可能只是第一次检查并暴露了更早产生的错误值，应向前追踪输入和调用者。

5. 判断：遇到 `ModuleNotFoundError`，直接运行 `pip install` 是最可靠的第一步。

   **参考答案：**错误。应先确认当前 Python、pip 路径、导入名称和项目要求，避免把包安装到另一套环境。

6. 判断：把 batch size 降低后能运行，就证明模型和实验已经正确。

   **参考答案：**错误。它可能只证明显存压力降低或触发路径改变；仍需检查 shape、结果、训练动态和实验意义。

7. 判断：`assert` 只能用于学习代码，科研项目中不应该使用。

   **参考答案：**错误。断言适合检查必须成立的前置条件和不变量，能让数据、shape、device 等问题更早暴露。

### Level 3：读错误与代码

8. 看到 `FileNotFoundError: data/train.json`，为什么要先检查当前工作目录？

   **参考答案：**相对路径通常相对于进程当前工作目录，而不是相对于 Python 文件位置。启动目录不对就会导致同一段代码找不到文件。

9. 下面代码最可能帮助你发现什么问题？

   ```python
   print(type(x), repr(x))
   assert x is not None
   ```

   **参考答案：**它可以发现变量是 `None`、空字符串、类型错误或包含不明显字符，并在后续深层调用前立即失败。

10. 解释下面三项在 AI Debug 中分别检查什么：`shape`、`dtype`、`device`。

    **参考答案：**`shape` 检查各维度的数量和语义；`dtype` 检查整数/浮点等数据类型是否适合操作；`device` 检查 Tensor 和模型是否位于兼容的 CPU/GPU 设备。

11. 下面的矩阵错误提示了什么？

    ```text
    mat1 and mat2 shapes cannot be multiplied (32x512 and 768x256)
    ```

    **参考答案：**矩阵乘法的内部维度 512 和 768 不一致。通常表示上游特征维度与当前线性层期望的输入维度不匹配，需要追踪模型配置、checkpoint 和前一层输出。

### Level 4：应用

12. 设计一个从 `import` 到一次参数更新的最小验证流程。

    **参考答案：**确认环境和入口 → 导入关键包 → 读取一个样本 → 组成一个 batch → 打印并断言 shape/dtype/device → 完成一次 forward → 计算有限的标量 loss → `backward()` → 检查梯度 → `optimizer.step()`。

13. 为什么“先让小数据集过拟合”能帮助 Debug？

    **参考答案：**小数据集让问题变得可控。如果模型连少量样本都无法拟合，可能存在标签、数据预处理、loss、梯度或模型连接问题；它是逻辑检查，不是最终性能证明。

14. 你发现代码修改后结果完全没变。列出至少三个检查方向。

    **参考答案：**确认运行的代码文件和 `__file__`；检查 Git diff 和实际 commit；打印最终配置确认参数生效；确认修改分支确实执行；检查是否使用旧安装包、缓存或固定随机种子。

### Level 5：Debug 场景

15. 训练第一步 loss 就是 `NaN`。请给出合理排查顺序。

    **参考答案：**先检查输入是否含 NaN/Inf，再检查输入范围和 dtype；然后检查模型输出、loss 输入和 loss 本身；定位第一次出现非有限值的操作，最后再考虑学习率、初始化或混合精度。

16. CPU 可以运行但 GPU 报 device 错误。你会检查哪些对象？

    **参考答案：**检查模型参数、输入、标签、隐藏状态和其他参与计算的 Tensor 的 `device`；确认 GPU 是否可见、PyTorch 构建是否匹配，并用最小 forward 逐步恢复 GPU。

### Level 6：综合科研场景

17. 一个 World Model 仓库启动训练时报 `KeyError: 'context_length'`。请说明判断过程。

    **参考答案：**先确认实际读取的配置文件和配置内容，再检查键名拼写、默认配置与命令行覆盖关系，以及代码版本是否与配置文件匹配。不要直接把异常捕获后设成任意默认值，因为 context length 可能改变模型输入和实验定义。

18. 实验没有报错，但 validation 指标远低于预期。请设计一个排查计划。

    **参考答案：**先固定 commit、环境和数据版本；检查验证阶段是否调用 `model.eval()` 和 `no_grad()`；人工检查样本、标签和预处理；用小数据集过拟合；对照 train/eval 的输出和指标实现；检查数据划分、标签错位、checkpoint 加载和配置参数，逐层排除。

19. 某次运行失败，第二次运行成功。为什么不能只把第一次当作偶发故障？

    **参考答案：**间歇性失败可能由随机数据、并发、显存残留、网络/挂载、未固定随机种子或竞态条件造成。应记录两次环境、命令、输入、随机种子、资源状态和完整日志，努力建立稳定复现条件。

20. 设计一个“保留证据的干净重试”流程，适用于你无法解释的训练错误。

    **参考答案：**保存完整 traceback、启动命令、配置、环境版本、GPU 信息、commit 和日志；复制或冻结当前状态；建立最小运行入口，只跑一个样本和 batch；按环境、路径、数据、shape/device、forward/loss/backward 分层验证；每次只做一个可逆改动；最后记录根因、修复和回归验证结果。

## 15. 掌握度检查（附答案）

学完本篇后，先尝试独立回答每个问题，再展开对应答案。

1. Debug 与“把报错修掉”有什么区别？

   **答案：**Debug 还要求理解现象、定位根因、验证修复并确认没有破坏其他行为；仅让程序暂时不报错可能掩盖问题。

2. 读 Python traceback 时，应该关注哪些部分？

   **答案：**关注调用路径、最接近业务代码的文件和行号、最下面的异常类型与消息，以及出错行收到的输入值。

3. 为什么要先写“期望结果”和“实际结果”？

   **答案：**两者的差异把模糊的“跑不对”转化为可验证问题，例如期望 `[B,C,H,W]`、实际缺少 batch 维。

4. Debug 的错误分层包括哪些主要层级？

   **答案：**启动、环境/依赖、路径/配置/数据、shape/dtype/device、模型/loss/梯度，以及最终结果可信度。

5. 为什么要先稳定复现，再修改代码？

   **答案：**如果问题不稳定，就无法判断某次修改是否真正解决了它，也可能把随机现象误判成代码原因。

6. `print`、日志、`assert` 和断点分别适合什么场景？

   **答案：**`print` 适合快速观察；日志适合持续、结构化记录；`assert` 适合检查必须成立的条件；断点适合暂停现场并交互查看变量和调用栈。

7. `FileNotFoundError` 为什么经常与工作目录有关？

   **答案：**相对路径相对于进程当前工作目录解析，而不一定相对于代码文件所在目录；从不同目录启动会得到不同的实际路径。

8. 如何排查 `ModuleNotFoundError` 而不盲目重装？

   **答案：**检查当前环境、`sys.executable`、`python -m pip --version`、`python -m pip show 包名`、导入名称和仓库依赖声明。

9. AI 代码中为什么要同时检查 shape、dtype 和 device？

   **答案：**它们分别描述数据结构、数据类型和计算位置；任何一项不匹配都可能造成直接报错或不易察觉的错误结果。

10. 怎样定位 loss 第一次变成 NaN 的位置？

    **答案：**从输入、模型输出、loss 输入到 loss 逐层使用 `torch.isfinite` 检查，并记录数值范围，找到第一次出现非有限值的操作。

11. 为什么小数据集过拟合是重要的 Debug 手段？

    **答案：**它能在低复杂度条件下检查数据、标签、模型、loss 和梯度是否基本连通；连小数据都拟合不了时，不宜先扩大训练规模。

12. 训练和评估阶段为什么要注意 `train()` 与 `eval()`？

    **答案：**Dropout、BatchNorm 等层在两种模式下行为不同；模式错误会导致指标不可信，即使程序不报错。

13. CUDA OOM 时，降低 batch size 说明了什么？

    **答案：**它可以验证显存压力是否是主要因素，但不等于最终方案，也可能改变训练动态；应记录并进一步分析模型、输入和显存使用。

14. 代码修改后结果不变时，为什么要检查 `__file__` 和 Git commit？

    **答案：**可能运行的不是刚修改的文件或代码版本，尤其是存在多个仓库副本、旧安装包或不同工作目录时。

15. 如何判断一个“修复”是否真的完成？

    **答案：**原错误消失、相关最小流程通过、原本正常的功能仍正常、输出符合预期，并且根因和预防措施已经被记录。
