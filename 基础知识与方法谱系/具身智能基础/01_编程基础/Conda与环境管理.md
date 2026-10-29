# Conda 与 Python 环境管理

这篇笔记解决一个很实际的问题：为什么同一台电脑或服务器上，不同 AI 项目经常需要不同版本的 Python、PyTorch、CUDA 相关库，以及怎样让一个项目稳定地运行起来。

在科研代码中，环境管理不是“安装几个包”的小事。它决定了你运行的 `python` 到底是谁、`pip` 把包安装到了哪里、项目能否复现，以及报错时应该从哪里开始排查。

> [!NOTE]
> 本篇聚焦“让 Python/AI 项目拥有可确认、可复现的运行环境”。Python 语法见 [[Python基础]]；命令行细节见 [[Linux与Shell]]；版本控制见 [[Git]]；代码报错的通用排查见 [[Debug与工程基础]]；NumPy 和 PyTorch 的数组/张量知识分别见 [[NumPy]]、[[PyTorch]]。

## 0. 学习目标

学完以后，应该能够：

- 用自己的话解释解释器、环境、package、dependency、channel、激活和 PATH。
- 判断当前终端使用的是哪个 Python，以及当前 `pip` 是否属于同一个环境。
- 创建、激活、验证、删除一个 Conda 环境。
- 读懂 `conda create -n project python=3.10`、`pip install -r requirements.txt` 等常见命令。
- 区分 Conda 环境、Python 包、项目目录和代码仓库。
- 按“仓库 → 环境 → 依赖 → 入口脚本 → 训练/评估”的顺序启动 AI research repository。
- 根据错误现象判断是环境没激活、包没装对、版本冲突、路径问题，还是代码本身的问题。
- 导出环境信息，为以后复现实验留下证据。

## 1. 为什么需要环境管理？

假设你有两个项目：

```text
项目 A：Python 3.10，较新的 PyTorch
项目 B：Python 3.8，旧版 PyTorch 和旧版依赖
```

如果所有项目都把包安装到同一个 Python 中，升级项目 A 可能破坏项目 B。常见结果包括：

- `import` 失败：代码找不到某个包。
- 版本冲突：A 需要 `package>=2`，B 只能使用 `package<2`。
- 命令和代码使用了不同的 Python：你明明安装成功，运行时却仍然报错。
- CPU/GPU 或 CUDA 相关依赖不匹配。
- 今天能跑，换到服务器或同事电脑就不能跑。

因此引入环境隔离：每个项目拥有一组相对独立的解释器和依赖。切换项目时切换环境，而不是不断修改同一个“全局 Python”。

## 2. 核心概念

### 2.1 Python 解释器（Python Interpreter）

**直觉**：解释器是“真正执行 Python 文件的程序”。

**正式含义**：它读取 Python 源代码，把代码转换成计算机可以执行的操作。代码/论文中通常直接写 `python`，但 `python` 只是一个命令名，最终必须解析到某个具体文件。

例如，下面两条命令可能调用不同的解释器：

```bash
python train.py
python3 train.py
```

在 Windows 上也可能是 `python.exe`；在 Linux 服务器上通常是 `.../bin/python`。同名不代表是同一个文件。

### 2.2 Python 环境（Environment）

**直觉**：环境是一套“运行某个项目所需的工具和包”。

它通常包括：

```text
一个 Python 解释器
一组已安装的 Python packages
包的版本信息
与这些包相关的命令
```

环境不是项目目录，也不是虚拟机。项目代码仍然在仓库目录中；环境只是运行这些代码时使用的工具集合。

### 2.3 虚拟环境（Virtual Environment）

**直觉**：在同一台机器上为不同项目划出彼此隔离的 Python 工作空间。

它不一定是完整的操作系统，也不一定隔离显卡驱动；它主要隔离 Python 解释器、Python 包和相关命令。Conda environment 是实现这种隔离的一种方式，Python 自带的 `venv` 是另一种方式。

### 2.4 包（Package）与依赖（Dependency）

**包 / package**：可以被 Python 导入并复用的一组代码，例如 `numpy`、`torch`。

**依赖 / dependency**：一个项目正常运行所依赖的外部软件或包。例如训练脚本 `import torch`，那么 PyTorch 就是该项目的依赖；它还可能依赖 NumPy、配置库或图像库。

包是“被安装的东西”；依赖是“项目与它之间的需要关系”。一个项目通常有很多直接依赖，而直接依赖还可能带来间接依赖。

### 2.5 Conda、pip 与 channel

**Conda** 是环境与软件包管理工具。它可以创建环境，也可以安装一部分 Python 包和非 Python 依赖。

**pip** 是 Python 包安装工具，主要从 Python Package Index（通常简称 PyPI）等来源安装 Python 包。

**channel** 是 Conda 查找软件包的来源地址或仓库。不同 channel 的包构建方式、版本和兼容性可能不同。当前阶段知道“来源会影响可获得版本和兼容性”即可。

一个简单判断：

```text
conda：负责环境，也能装包
pip：负责 Python 包
项目：声明自己需要哪些依赖
```

不要把 Conda、pip、Python、PyTorch 当成同一个东西。

### 2.6 激活（activate）与 PATH

**激活**不是“启动了一个服务器”。它是在当前终端中调整环境变量，尤其是 `PATH`，使得输入 `python` 时优先找到目标环境中的解释器和命令。

```text
输入 python
    ↓
终端按照 PATH 搜索可执行文件
    ↓
优先找到已激活环境中的 python
```

关闭终端或执行 `conda deactivate` 后，当前终端不再优先使用该环境。环境本身通常仍然存在。

## 3. Conda 的整体工作流

```text
获取 repository
    ↓
阅读 README / environment.yml / requirements.txt
    ↓
创建独立环境
    ↓
激活环境
    ↓
安装项目依赖
    ↓
验证 python、pip、关键包
    ↓
运行入口脚本
    ↓
记录环境并排查错误
```

创建环境并不等于安装项目依赖。激活环境也不等于代码一定能运行；它只让当前终端开始使用那套环境。

## 4. 从零开始：创建并验证环境

下面以名为 `jepa`、Python 3.10 的环境为例。环境名只是标签，可以改成项目名。

```bash
conda create -n jepa python=3.10
```

逐段解释：

| 部分 | 含义 |
|---|---|
| `conda` | 调用 Conda 程序 |
| `create` | 创建一个新环境 |
| `-n` | `--name` 的短参数，表示接下来给环境命名 |
| `jepa` | 环境名，不是 Python 包，也不是仓库目录 |
| `python=3.10` | 请求在环境中安装 Python 3.10 系列 |

执行后，Conda 会解析依赖、下载所需文件，并在自己的环境目录中建立解释器和包索引。它没有做这些事：没有下载 GitHub 仓库，没有安装项目全部依赖，没有运行训练，也没有保证 GPU 可用。

激活并验证：

```bash
conda activate jepa
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

观察重点：

- `python --version` 应接近 `Python 3.10.x`。
- `sys.executable` 应指向 `jepa` 环境中的 Python，而不是系统 Python。
- `python -m pip --version` 显示的路径应与当前 Python 环境对应。

推荐使用 `python -m pip`，因为它明确表示“用这个 Python 去运行 pip”，比单独输入 `pip` 更不容易串环境。

列出和离开环境：

```bash
conda env list
conda deactivate
```

`conda env list` 中带 `*` 的通常是当前激活环境。`conda deactivate` 只改变当前终端的使用状态，不会删除环境。

删除环境是有破坏性的操作，确认不再需要后再执行：

```bash
conda env remove -n jepa
```

## 5. 安装依赖：Conda 与 pip 怎么配合？

先进入项目目录，再阅读它提供的依赖文件：

```bash
cd path/to/repository
Get-ChildItem                 # PowerShell
ls                            # Linux/macOS 常见写法
```

常见文件：

| 文件 | 通常表示 |
|---|---|
| `environment.yml` | Conda 环境及其依赖声明 |
| `requirements.txt` | pip 依赖列表 |
| `pyproject.toml` | 现代 Python 项目配置和依赖声明 |
| `setup.py` | 较老但仍常见的安装配置 |

如果仓库明确提供命令，优先按 README 执行，因为 PyTorch/CUDA 等安装方式可能依赖操作系统和显卡条件。一个常见流程是：

```bash
conda activate jepa
python -m pip install -r requirements.txt
```

`-r` 表示从文件读取依赖，而不是把 `requirements.txt` 当作一个包名。安装完成后，至少验证关键依赖：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

这里 `import torch` 只验证 PyTorch 能否被当前解释器导入；`torch.cuda.is_available()` 返回 `True` 才说明当前 PyTorch 进程看到了可用 CUDA，返回 `False` 不一定就是 Conda 错误，还可能是没有 GPU、驱动问题或仓库安装的是 CPU 版本。CUDA 细节见项目文档和 [[PyTorch]]，不要只凭一个布尔值猜原因。

## 6. 完整 AI research repository 场景

一个典型项目可以这样理解：

```text
GitHub repository
    ↓ clone 到本地/服务器
进入项目目录
    ↓
创建并激活 Conda 环境
    ↓
安装 requirements / environment.yml 中的依赖
    ↓
读取 config，准备 dataset
    ↓
运行 train.py 或 scripts/train.sh
    ↓
保存 checkpoint，运行 evaluation
```

示意命令：

```bash
git clone <repository-url>
cd <repository-directory>
conda create -n project python=3.10
conda activate project
python -m pip install -r requirements.txt
python train.py --config configs/base.yaml
```

逐步判断：

1. `git clone` 只取得代码和仓库历史，不会自动安装依赖。Git 见 [[Git]]。
2. `cd` 只改变当前命令行所在目录，不会改变 Python 环境。目录与环境是两件事。
3. `conda create` 建立解释器和环境基础，不会自动理解项目代码。
4. `pip install` 根据依赖文件安装包；成功信息不代表训练逻辑正确。
5. `python train.py` 才是让当前解释器加载项目代码的动作。脚本、参数和配置属于项目工程结构，见 [[Python基础]] 与 [[Debug与工程基础]]。

真实研究中还要记录：仓库 commit、Python 版本、PyTorch 版本、GPU/驱动信息、数据路径和启动命令。这些信息共同决定实验能否复现。

## 7. 环境文件与可复现

查看当前环境中的包：

```bash
conda list
python -m pip list
```

导出环境：

```bash
conda env export > environment.yml
python -m pip freeze > requirements-lock.txt
```

这两个命令会把输出重定向到文件。它们记录的是当前环境状态，不等于自动解决所有跨机器差异。尤其是 GPU、系统驱动、操作系统和本地路径，可能不会被普通 Python 依赖文件完整描述。

恢复 Conda 环境的常见方式：

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

在团队项目中，提交环境文件前应检查其中是否包含本机绝对路径、无关的本地包或过度具体的平台信息。

## 8. 常见错误与 Debug

### 8.1 `conda: command not found` 或无法识别 `conda`

- **现象**：终端找不到 `conda`。
- **原因**：Conda 未安装、没有初始化当前 shell，或 PATH 未配置。
- **判断**：运行 `conda --version`；确认你打开的是已初始化的终端。
- **处理**：使用正确的 Anaconda/Miniconda 终端，或按 Conda 官方方式初始化对应 shell，然后重新打开终端。
- **为什么**：系统必须先找到 Conda 可执行文件，才能谈创建环境。Shell 基础见 [[Linux与Shell]]。

### 8.2 激活后 `python --version` 不对

- **现象**：激活 `jepa` 后仍显示系统 Python。
- **原因**：激活没有生效、终端缓存/配置异常，或实际调用的是别的命令。
- **判断**：运行 `python -c "import sys; print(sys.executable)"` 和 `conda env list`。
- **处理**：重新打开初始化后的终端，重新 `conda activate jepa`；确认环境确实存在。
- **为什么**：环境名只是选择器，真正重要的是解释器路径。

### 8.3 `pip install` 成功，但 `import` 失败

- **现象**：安装输出成功，运行代码却 `ModuleNotFoundError`。
- **原因**：安装时的 pip 和运行时的 python 不属于同一环境。
- **判断**：分别运行 `where.exe python`（Windows）或 `which python`（Linux），以及 `python -m pip --version`。
- **处理**：先激活正确环境，再使用 `python -m pip install package-name`，最后用同一个 `python` 测试导入。
- **为什么**：系统可能同时存在多个 Python；单独的 `pip` 可能指向另一套解释器。

### 8.4 依赖冲突或安装解析失败

- **现象**：Conda/pip 报版本冲突、无法解析依赖。
- **原因**：项目要求的版本组合不存在，或混用了不兼容来源。
- **判断**：记录完整错误、Python 版本、操作系统和已安装包；查看 README 的固定版本。
- **处理**：从干净环境重新按项目文档安装；不要看到一个包缺失就随意升级所有包。
- **为什么**：依赖不是孤立的，升级一个包可能改变它的间接依赖。更系统的错误记录见 [[Debug与工程基础]]。

### 8.5 `FileNotFoundError` 或数据集找不到

- **现象**：环境和 `import` 都正常，但训练启动时找不到数据。
- **原因**：当前工作目录、配置中的相对路径或数据未下载/挂载。
- **判断**：打印当前目录，检查 config 中的数据路径和文件是否真实存在。
- **处理**：进入仓库根目录运行，或按项目要求修改配置/准备数据；不要把“找不到数据”误判成 Conda 问题。

### 8.6 CUDA 不可用

- **现象**：`torch.cuda.is_available()` 为 `False`，或训练时报 CUDA 错误。
- **原因**：可能没有 GPU、驱动不可用、PyTorch 构建不匹配，或服务器资源未分配。
- **判断**：先区分“当前机器没有 GPU”和“有 GPU 但当前进程看不到”；查看项目要求、PyTorch 版本和服务器分配状态。
- **处理**：按仓库/官方安装矩阵重装匹配版本，或先使用 CPU 配置验证代码流程。
- **为什么**：Conda 环境不能替代操作系统级 GPU 驱动；环境问题与硬件/驱动问题要分层排查。

## 9. 容易混淆的概念

| 概念 | 是什么 | 不是什么 |
|---|---|---|
| Python | 一种语言，也常指解释器程序 | 不是 Conda 环境 |
| Conda | 环境和包管理工具 | 不是项目代码 |
| pip | Python 包安装工具 | 不是 Python 解释器 |
| 环境 | 一套解释器、包和相关命令 | 不是仓库目录 |
| package | 可安装/导入的代码单元 | 不等于整个项目 |
| dependency | 项目运行所需要的外部组件 | 不只是“随便装的包” |
| 激活 | 让当前终端优先使用某环境 | 不会自动安装依赖 |
| `import` 成功 | 当前 Python 找到了可导入模块 | 不代表训练一定正确 |
| `environment.yml` | Conda 环境声明/导出文件 | 不保证包含驱动和数据 |

## 10. 与其他笔记的关系

本篇重点负责：环境隔离、Conda/pip 基础、解释器与 PATH、依赖安装、验证、导出和科研仓库启动。

- 命令行、路径、shell 差异：[[Linux与Shell]]
- Python 语法、模块、脚本和项目结构：[[Python基础]]
- clone、commit 和仓库版本：[[Git]]
- 错误分类、日志和通用排查：[[Debug与工程基础]]
- 数组与 `shape`：[[NumPy]]
- Tensor、CUDA、训练与评估：[[PyTorch]]

本篇故意没有系统展开 Python 语法、Linux 命令、Git 工作流、CUDA 驱动原理、NumPy 数组运算或 PyTorch Tensor/autograd；这里只解释启动和验证 AI 项目所必需的最小部分。

## 11. 本篇知识地图

```text
项目要运行
├── 找到正确的 Python 解释器
├── 用环境隔离不同项目
│   └── conda create / conda activate
├── 安装依赖
│   ├── environment.yml
│   └── requirements.txt + python -m pip
├── 验证
│   ├── python 版本与路径
│   ├── pip 是否属于同一 Python
│   └── 关键包 / CUDA 是否可用
├── 运行 repository
│   ├── 进入正确目录
│   ├── 读取 config 与数据路径
│   └── train / evaluation
└── 记录与 Debug
    ├── 版本、commit、硬件
    └── 按现象分层排查
```

## 12. 快速查询表

| 我要做什么 | 命令 | 作用 |
|---|---|---|
| 查看 Conda | `conda --version` | 确认 Conda 可调用 |
| 创建环境 | `conda create -n name python=3.10` | 创建指定 Python 环境 |
| 激活环境 | `conda activate name` | 当前终端使用该环境 |
| 查看环境 | `conda env list` | 列出环境并标出当前环境 |
| 验证 Python | `python --version` | 查看版本 |
| 验证路径 | `python -c "import sys; print(sys.executable)"` | 查看实际解释器 |
| 安装依赖 | `python -m pip install -r requirements.txt` | 用当前 Python 安装依赖 |
| 查看包 | `conda list` / `python -m pip list` | 查看已安装包 |
| 导出环境 | `conda env export > environment.yml` | 保存环境描述 |
| 离开环境 | `conda deactivate` | 当前终端退出环境 |

## 13. 练习题（题目下附参考答案）

### Level 1：概念

1. 用自己的话解释：Python 解释器、环境、package、dependency 分别是什么？

   **参考答案：**解释器是执行 Python 的程序；环境是一套解释器与包；package 是可安装/导入的代码单元；dependency 是项目运行需要的外部组件。

2. 为什么两个研究项目最好不要共用一个不断修改的全局 Python？

   **参考答案：**不同项目可能需要互相冲突的 Python 或包版本。隔离环境可以让一个项目的升级不破坏另一个项目。

3. `conda activate jepa` 改变了什么？它没有改变什么？

   **参考答案：**它调整当前终端的环境变量，使命令优先找到 `jepa` 中的 Python 和工具；它不会下载仓库、安装全部依赖或启动训练。

### Level 2：判断

4. 判断：创建 Conda 环境后，仓库中的所有依赖已经自动安装。

   **参考答案：**错误。创建环境通常只安装指定 Python 和基础依赖；项目依赖仍需按 README、`environment.yml` 或 `requirements.txt` 安装。

5. 判断：`pip install` 成功，就能证明之后运行项目的 `python` 一定能 `import` 该包。

   **参考答案：**错误。安装时的 pip 可能属于另一套 Python；应检查 `python -m pip --version`、解释器路径，并用同一个 Python 做 `import` 验证。

6. 判断：`torch.cuda.is_available()` 为 `False` 时，问题一定是 Conda 环境损坏。

   **参考答案：**错误。还可能是没有 GPU、驱动不可用、PyTorch 构建不匹配或服务器没有分配 GPU。

### Level 3：读命令

7. 逐段解释 `conda create -n robotlab python=3.10`，并说明执行后没有完成哪些事情。

   **参考答案：**调用 Conda，创建名为 `robotlab` 的环境，并请求 Python 3.10。它没有 clone 仓库、安装项目全部依赖、准备数据或启动训练。

8. 为什么推荐 `python -m pip install -r requirements.txt`，而不是盲目使用 `pip install -r requirements.txt`？

   **参考答案：**`python -m pip` 明确指定由当前 `python` 运行 pip，可以降低多个 Python 并存时的串环境风险。

9. 下面三条命令分别验证什么？

   ```bash
   python --version
   python -c "import sys; print(sys.executable)"
   python -m pip --version
   ```

   **参考答案：**第一条看 Python 版本；第二条看实际解释器文件路径；第三条看与当前 Python 绑定的 pip 及其安装位置。三者要共同判断环境是否正确。

### Level 4：场景与应用

10. 你拿到一个 World Model 仓库，包含 `environment.yml` 和 `train.py`。写出从进入仓库到启动训练的合理顺序，并说明每一步的目的。

   **参考答案：**进入仓库 → 阅读 README 和依赖文件 → 创建环境 → 激活 → 安装依赖 → 验证 Python、关键包和必要的 CUDA → 按配置准备数据 → 运行 `train.py`。顺序体现了先准备运行条件、再执行代码。

11. 一个项目需要 Python 3.8，另一个项目需要 Python 3.10。你会如何组织环境？

   **参考答案：**创建两个独立环境，例如 `project-a` 使用 Python 3.8，`project-b` 使用 Python 3.10；进入对应项目时激活对应环境。

12. 你准备把实验交给同学复现，至少记录哪些环境和项目状态信息？

   **参考答案：**至少记录仓库 commit、Python 版本、关键包版本、GPU/驱动信息、数据路径/版本、配置文件和完整启动命令。

### Level 5：Debug

13. 激活 `jepa` 后，`python --version` 显示正确，但 `import torch` 失败。请给出排查顺序。

   **参考答案：**Python 版本正确只说明版本正确，不说明 PyTorch 已安装。排查顺序：确认 `sys.executable` → `python -m pip show torch` 或 `python -m pip list` → 读取完整导入错误 → 对照仓库要求安装匹配版本 → 再次用当前 Python 导入。

14. `python -m pip install numpy` 显示成功，但 `python -c "import numpy"` 失败。你会先检查什么，而不是先重装所有东西？

   **参考答案：**先检查 `sys.executable`、`python -m pip --version`、`python -m pip show numpy` 和完整错误信息，确认安装位置与导入解释器一致；再判断是否存在包损坏或版本冲突。

15. PyTorch 可以导入，但训练启动时报 `FileNotFoundError`。为什么这通常不是 Conda 安装问题？

   **参考答案：**`FileNotFoundError` 表示代码访问的路径不存在或当前工作目录不对。应检查当前目录、配置中的相对路径、数据是否准备好，而不是先重装 Conda。

### Level 6：综合

16. 设计一个“干净重试”方案：某仓库安装过程中发生多个版本冲突。要求说明你会保留哪些证据、如何建立新环境、如何验证，以及哪些操作不能随意做。

   **参考答案：**先保存完整错误、README、依赖文件、Python/系统信息和当前 commit；不要继续污染旧环境；创建新的项目环境，按文档选择一致的安装来源，先验证 Python 与关键包，再运行最小入口；若仍失败，区分依赖解析、路径、数据、GPU 和代码逻辑问题，逐层排查。不能随意升级全部包或删除环境而不保留证据。

## 14. 掌握度检查（附答案）

学完本篇后，先尝试独立回答每个问题，再展开对应答案。

1. Python 解释器和 Python 环境有什么区别？

   **答案：**解释器是实际执行 Python 代码的程序；环境是包含解释器、已安装包、版本信息和相关命令的一整套运行空间。

2. 为什么研究项目需要环境隔离？

   **答案：**不同项目可能需要不同 Python 或依赖版本。隔离可以避免项目之间互相破坏，也便于复现实验。

3. Conda 和 pip 分别解决什么问题？

   **答案：**Conda 主要负责创建和管理环境，也能安装部分软件包；pip 主要负责安装 Python 包。两者不是同一个工具。

4. package 和 dependency 为什么不是同义词？

   **答案：**package 是一个可安装或导入的代码单元；dependency 强调“项目需要它才能运行”的关系。一个 package 可以成为某个项目的 dependency。

5. `conda create -n name python=3.10` 每一部分是什么意思？

   **答案：**`conda` 调用工具；`create` 创建环境；`-n` 表示指定名称；`name` 是环境名；`python=3.10` 请求安装 Python 3.10。它不会自动下载仓库或安装全部项目依赖。

6. 激活环境时 PATH 大致发生了什么变化？

   **答案：**当前终端的 PATH 被调整，使目标环境中的 Python 和命令优先于系统中其他同名命令被找到。

7. 如何确认当前 `python` 的真实路径？

   **答案：**运行 `python -c "import sys; print(sys.executable)"`；Windows 还可以运行 `where.exe python`，Linux/macOS 常用 `which python`。

8. 如何确认 pip 和当前 Python 属于同一个环境？

   **答案：**运行 `python -m pip --version`，查看它显示的 Python 环境路径；不要只看单独的 `pip --version`。

9. 创建环境后为什么还不能直接认为项目可以运行？

   **答案：**创建环境通常只准备 Python 和基础组件；项目依赖、数据、配置、工作目录、GPU 条件和入口参数仍可能没有准备好。

10. `environment.yml` 和 `requirements.txt` 的职责有什么不同？

    **答案：**`environment.yml` 通常描述 Conda 环境及其依赖；`requirements.txt` 通常列出 pip 要安装的 Python 包。具体以仓库 README 为准。

11. 为什么 `import` 成功不代表训练逻辑正确？

    **答案：**`import` 只说明解释器能找到并加载包；数据路径、模型配置、张量 shape、训练参数和代码逻辑仍可能出错。

12. CUDA 不可用时，如何区分环境、驱动、硬件和资源分配问题？

    **答案：**先确认机器是否有 GPU、服务器是否分配 GPU，再检查驱动和 PyTorch 构建/版本，最后检查代码使用的设备配置。不能仅凭 `False` 判定是 Conda 损坏。

13. 一个 AI repository 从 clone 到 train 通常经历哪些阶段？

    **答案：**clone 仓库 → 进入项目目录 → 阅读 README 和依赖文件 → 创建并激活环境 → 安装依赖 → 验证解释器与关键包 → 准备数据和配置 → 运行训练入口脚本。

14. 如何记录实验环境以帮助复现？

    **答案：**记录仓库 commit、Python 和关键包版本、GPU/驱动信息、数据版本和路径、配置文件、环境导出文件以及完整启动命令。

15. 遇到 `ModuleNotFoundError` 时，你的第一轮排查顺序是什么？

    **答案：**先确认当前环境是否激活，再检查 `sys.executable` 和 `python -m pip --version` 是否对应同一环境，接着用 `python -m pip show 包名` 查看安装情况，最后根据完整错误判断包缺失、名称差异或版本冲突。
