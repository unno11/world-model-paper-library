# Linux 与 Shell：AI 科研命令行基础

很多 AI 项目不是在图形界面里点击按钮运行的，而是在本地终端、远程服务器或集群中执行命令。你会看到：

```bash
git clone <repository-url>
cd project
conda activate project
python train.py --config configs/base.yaml
```

如果不了解命令行，就很难判断自己在哪个目录、命令操作了什么、程序是否仍在运行、日志保存在哪里，以及为什么同一条命令在 Windows 和 Linux 上表现不同。

本篇学习的是“能读懂并安全使用 AI 项目常见 Shell 操作”的基础，不以培养系统管理员为目标。

> [!NOTE]
> 本篇负责 Shell、终端、路径、文件操作、环境变量、管道、进程和基础脚本。Conda 环境见 [[Conda与环境管理]]；Git 版本管理见 [[Git]]；错误定位见 [[Debug与工程基础]]；Python 语法见 [[Python基础]]。

## 0. 学习目标

学完以后，应该能够：

- 解释终端、Shell、命令、参数、选项和当前工作目录。
- 区分 Windows PowerShell 与 Linux/macOS Shell 的常见写法。
- 使用 pwd、cd、ls/Get-ChildItem、mkdir、cp/Copy-Item 等命令安全地浏览项目。
- 理解绝对路径、相对路径、文件名、目录和路径分隔符。
- 查看环境变量，知道 PATH 为什么会影响命令查找。
- 使用管道和重定向理解“一个命令的输出如何交给下一个命令”。
- 查看进程、停止失控任务、使用后台运行和日志。
- 读懂 AI 项目常见的 train.sh、评估命令和多行 Shell 脚本。
- 在执行删除、覆盖、后台训练等操作前先确认目标和影响。

## 1. 为什么 AI 研究需要命令行？

真实仓库常见的启动过程：

```text
服务器登录
    ↓
进入项目目录
    ↓
确认 Git commit
    ↓
激活 Conda 环境
    ↓
准备数据和配置
    ↓
启动训练
    ↓
查看日志和 GPU 进程
    ↓
保存 checkpoint 和结果
```

命令行提供了三个重要能力：

1. **精确**：命令可以明确写出目录、参数和配置文件。
2. **可重复**：可以保存完整命令，之后再次执行。
3. **可远程**：通过 SSH 登录服务器后，不需要图形界面也能操作。

但命令行也有风险：当前目录错了，命令可能作用于另一份代码；路径错了，程序可能读到错误数据；删除和覆盖命令可能造成不可逆损失。因此本篇始终强调“先查看，再操作，再验证”。

## 2. 核心概念

### 2.1 终端（Terminal）

终端是你输入命令、查看输出的界面。它本身通常不负责理解命令，而是把输入交给 Shell。

### 2.2 Shell

Shell 是命令解释器。它读取命令、解析参数、查找程序，并启动程序。

常见环境：

| 环境 | 常见命令形式 |
|---|---|
| Linux/macOS Bash 或 Zsh | pwd、ls、cd、python |
| Windows PowerShell | Get-Location、Get-ChildItem、Set-Location |
| Windows 命令提示符 | cd、dir |

Linux 服务器上常见 Bash；Windows 上常见 PowerShell。命令含义相近，但参数、路径分隔符和通配符规则可能不同。

### 2.3 命令、参数与选项

命令可以看成“要求哪个程序做什么”。

```bash
python train.py --config configs/base.yaml
```

这里：

- python 是命令，调用 Python 解释器。
- train.py 是位置参数，告诉 Python 执行哪个脚本。
- --config 是选项名。
- configs/base.yaml 是选项的值。

选项通常以 - 或 -- 开头，但具体格式由程序决定。不要看到任何带短横线的内容就认为它有统一含义；要看该命令的帮助说明。

```bash
python train.py --help
git --help
```

PowerShell 中也可以运行 Get-Help，但外部程序的帮助格式由程序自己决定。

### 2.4 当前工作目录（Current Working Directory）

当前工作目录简称 cwd，是命令执行时默认所在的目录。

```bash
pwd
```

PowerShell：

```powershell
Get-Location
```

相对路径会从当前工作目录开始解析。例如当前目录是：

```text
D:\projects\world-model
```

访问 data/train.json，实际指向：

```text
D:\projects\world-model\data\train.json
```

如果你从 D:\projects 启动同一段代码，相对路径的含义就会变化。这是 AI 项目 FileNotFoundError 的常见原因，见 [[Debug与工程基础]]。

### 2.5 文件、目录与路径

- 文件是保存内容的对象，如 train.py、config.yaml。
- 目录是组织文件和子目录的容器。
- 路径是描述对象位置的文字。

路径有两种：

**绝对路径**从盘符或根目录开始，位置完整明确。

```text
D:\projects\world-model\configs\base.yaml
/home/user/projects/world-model/configs/base.yaml
```

**相对路径**相对于当前工作目录。

```text
configs/base.yaml
../data/train.json
./train.py
```

常见符号：

| 符号 | 含义 |
|---|---|
| . | 当前目录 |
| .. | 上一级目录 |
| / | Linux/macOS 的路径分隔符，也表示根目录的一部分 |
| \ | Windows 常见路径分隔符 |
| ~ | 当前用户主目录，在许多 Unix Shell 中可展开 |

代码中常用 pathlib 处理路径：

```python
from pathlib import Path

project = Path.cwd()
config = project / "configs" / "base.yaml"
print(config.resolve())
print(config.exists())
```

### 2.6 PATH 环境变量

环境变量是进程可以读取的一组键值信息。PATH 是其中非常重要的变量，它告诉 Shell 到哪些目录中查找可执行程序。

当你输入 python 时，Shell 会按照 PATH 中的目录顺序寻找 python。多个 Python 或 Git 并存时，PATH 顺序会影响实际调用哪个程序。

检查：

```bash
echo $PATH
which python
which git
```

PowerShell：

```powershell
$env:PATH
Get-Command python
Get-Command git
```

Conda 激活环境会调整 PATH，使环境中的 Python 优先被找到。详见 [[Conda与环境管理]]。

## 3. 文件系统的基本操作

### 3.1 查看目录

Linux/macOS：

```bash
ls
ls -la
```

PowerShell：

```powershell
Get-ChildItem
Get-ChildItem -Force
```

-la 通常表示显示详细信息和隐藏文件；PowerShell 的 -Force 有类似查看隐藏项目的作用。

先查看再操作：

```powershell
Get-ChildItem -LiteralPath "D:\projects\world-model"
```

LiteralPath 的意义是把字符串当作准确路径，避免通配符被误解释。

### 3.2 进入目录

```bash
cd project
cd ..
cd /home/user/project
```

PowerShell 也常用：

```powershell
Set-Location "D:\projects\world-model"
Set-Location ..
```

验证：

```bash
pwd
ls
```

进入仓库后先看 pwd 和目录内容，不要只凭提示符猜测位置。

### 3.3 创建目录

Linux/macOS：

```bash
mkdir -p outputs/2026-08-09
```

PowerShell：

```powershell
New-Item -ItemType Directory -Path ".\outputs\2026-08-09"
```

mkdir 创建目录。-p 表示必要时创建上层目录并在目录已经存在时不报错；PowerShell 使用 New-Item 更明确。

### 3.4 复制与移动

Linux/macOS：

```bash
cp configs/base.yaml configs/backup.yaml
mv old_name.py new_name.py
```

PowerShell：

```powershell
Copy-Item -LiteralPath ".\configs\base.yaml" -Destination ".\configs\backup.yaml"
Move-Item -LiteralPath ".\old_name.py" -Destination ".\new_name.py"
```

复制通常保留原文件；移动会改变文件位置或名称。涉及覆盖时要先确认目标是否存在。

### 3.5 删除：必须谨慎

Linux/macOS：

```bash
rm file.txt
rmdir empty_dir
```

PowerShell：

```powershell
Remove-Item -LiteralPath ".\file.txt"
Remove-Item -LiteralPath ".\empty_dir"
```

递归删除目录尤其危险。不要对不确定的路径使用宽泛通配符或递归删除。执行前：

1. 打印目标的绝对路径。
2. 列出目标内容。
3. 确认目标在预期项目目录内。
4. 确认没有需要保留的 checkpoint、数据或日志。
5. 优先采用可恢复方式。

本篇只要求知道删除存在风险；不要把命令行当作“无确认的清理工具”。

## 4. 查看和搜索文件

### 4.1 查看文本

Linux/macOS：

```bash
cat README.md
head -n 20 README.md
tail -n 50 train.log
```

PowerShell：

```powershell
Get-Content -LiteralPath ".\README.md"
Get-Content -LiteralPath ".\train.log" -Tail 50
```

大文件不要一次全部输出；先看头部、尾部或搜索关键字。

### 4.2 搜索文件名

如果安装了 ripgrep，推荐：

```bash
rg --files
rg --files -g "*.py"
```

PowerShell 也可以：

```powershell
Get-ChildItem -Recurse -File -Filter "*.py"
```

### 4.3 搜索文本

```bash
rg "learning_rate" .
rg "FileNotFoundError" logs/
```

PowerShell：

```powershell
Get-ChildItem -Recurse -File | Select-String -Pattern "learning_rate"
```

搜索时先限制目录和文件类型，避免把数据集、缓存和巨大日志全部扫一遍。

## 5. 管道、重定向与命令组合

### 5.1 管道（Pipe）

管道符号 | 把前一个命令的输出交给后一个命令。

```bash
git log --oneline | head -n 10
```

直觉：

```text
命令 A 的输出
    ↓ |
命令 B 继续筛选或处理
```

PowerShell 管道传递的通常是对象，因此可以继续按属性筛选：

```powershell
Get-ChildItem | Where-Object Length -gt 1000000
```

### 5.2 重定向（Redirection）

```bash
python train.py > train.log
python train.py >> train.log
```

> 表示覆盖写入；>> 表示追加写入。覆盖重定向前要确认文件可以被替换。

在 PowerShell 中也可以使用 > 和 >>，但输出格式和编码行为可能与 Linux 不同。

把错误也保存到日志是进阶用法，当前知道 stderr 可能与普通输出分开即可：

```bash
python train.py > train.log 2>&1
```

### 5.3 不要把多条命令当成一条

下面的命令依赖前一步成功：

```bash
cd project && python train.py
```

&& 表示前一条成功后才执行后一条。不同 Shell 对符号支持不同，PowerShell 常见写法也可能不同。复杂命令先拆开运行，更适合初学者 Debug。

## 6. 进程与长时间训练

### 6.1 什么是进程？

**进程 / process** 是一个正在运行的程序实例。执行 python train.py 后，操作系统会为它创建进程；进程有自己的内存、状态和进程号 PID。

### 6.2 查看进程

Linux：

```bash
ps aux | grep python
pgrep -af train.py
```

PowerShell：

```powershell
Get-Process python
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'"
```

### 6.3 停止进程

如果程序在当前终端运行，通常可用 Ctrl+C 请求中断。程序可能需要时间保存状态或清理资源。

Linux：

```bash
kill <PID>
```

PowerShell：

```powershell
Stop-Process -Id <PID>
```

不要先使用强制终止；先确认 PID 确实属于自己的训练任务。误杀别人的进程可能造成实验和数据损失。

### 6.4 后台运行和日志

Linux 服务器中常见：

```bash
nohup python train.py --config configs/base.yaml > logs/train.log 2>&1 &
tail -f logs/train.log
```

含义：

- nohup 让进程尽量不因终端断开而立即结束。
- > logs/train.log 保存普通输出。
- 2>&1 把错误输出合并到同一日志。
- & 请求后台运行。
- tail -f 持续查看日志末尾。

后台运行不是可靠的任务管理系统。长实验可以了解 tmux、作业调度器或项目提供的启动脚本，但目前知道它们存在即可。

## 7. Shell 脚本基础

### 7.1 为什么使用脚本？

把命令写入脚本可以：

- 记录完整启动方式。
- 减少手工输入错误。
- 让同一实验更容易重复。
- 集中设置环境变量和输出目录。

一个简单 Bash 脚本：

```bash
#!/usr/bin/env bash
set -e

CONFIG=configs/base.yaml
OUTPUT_DIR=outputs/baseline

mkdir -p "$OUTPUT_DIR"
python train.py --config "$CONFIG" --output-dir "$OUTPUT_DIR"
```

### 7.2 逐行理解

- 第一行选择解释器。
- set -e 表示命令失败时脚本退出，避免后续命令在错误状态上继续。
- CONFIG 和 OUTPUT_DIR 是 Shell 变量。
- $CONFIG 表示读取变量值。
- 双引号可以保护带空格的路径。
- mkdir -p 准备输出目录。
- 最后一行启动训练。

PowerShell 脚本使用 .ps1 文件，变量通常以 $ 开头：

```powershell
$config = ".\configs\base.yaml"
$outputDir = ".\outputs\baseline"

New-Item -ItemType Directory -Force -Path $outputDir
python .\train.py --config $config --output-dir $outputDir
```

### 7.3 脚本中的安全习惯

- 在脚本开始打印当前目录和关键参数。
- 使用明确的绝对/相对路径策略。
- 输出环境、Git commit 和配置路径。
- 对输入路径做存在性检查。
- 不要把密码、token 直接写入脚本。
- 训练脚本中避免默认覆盖旧输出。

## 8. 真实 AI 项目场景

```text
GitHub repository
    ↓ git clone
本地/服务器项目目录
    ↓ cd + pwd/ls
确认仓库位置
    ↓ conda activate
进入正确 Python 环境
    ↓ bash scripts/train.sh
启动训练
    ↓ logs/ + ps/nvidia-smi
观察输出和资源
    ↓ checkpoint/
保存模型和实验记录
```

一个排查顺序示例：

```bash
pwd
git status
git log -1 --oneline
python -c "import sys; print(sys.executable)"
test -f configs/base.yaml
python train.py --config configs/base.yaml
```

PowerShell 中相应检查：

```powershell
Get-Location
git status
git log -1 --oneline
python -c "import sys; print(sys.executable)"
Test-Path ".\configs\base.yaml"
python .\train.py --config ".\configs\base.yaml"
```

每一步都在验证不同层次：目录、代码版本、Python 环境、配置路径和入口脚本。

## 9. 常见错误与 Debug

### 9.1 cd 后仍然找不到文件

- **现象**：进入目录后，程序仍提示配置或数据不存在。
- **原因**：进入的不是预期目录，或相对路径相对于另一个 cwd。
- **判断**：运行 pwd/Get-Location、列出文件、打印 Path.cwd()。
- **处理**：进入仓库根目录，或在配置中使用明确的数据根目录。
- **为什么**：提示符和记忆不可靠，实际路径才是证据。

### 9.2 command not found / 无法识别命令

- **现象**：Shell 找不到 python、git、conda 或某个脚本。
- **原因**：程序未安装、PATH 未配置、环境未激活、命令名不同或当前 Shell 不支持该写法。
- **判断**：使用 which/Get-Command 和 --version。
- **处理**：确认安装和 PATH；按当前平台改用正确命令；Conda 问题见 [[Conda与环境管理]]。
- **为什么**：Shell 必须先找到可执行文件，才可能运行它。

### 9.3 脚本权限或换行问题

- **现象**：Linux 上脚本不能执行，或出现解释器/换行相关错误。
- **原因**：文件没有执行权限、脚本首行不正确，或 Windows 换行符造成兼容问题。
- **判断**：查看文件类型和权限，直接用 bash script.sh 试运行。
- **处理**：按项目要求处理权限和换行；不要把 Windows PowerShell 脚本直接当 Bash 执行。
- **为什么**：Shell 语法和平台不是完全通用的。

### 9.4 训练退出但没有日志

- **现象**：命令返回，输出目录为空。
- **原因**：入口函数没有执行、日志被重定向到其他位置、路径错误或脚本提前退出。
- **判断**：打印 cwd、输出目录、完整命令和退出码；查看日志头尾。
- **处理**：确认入口调用 main，确认输出目录和日志路径，按 [[Debug与工程基础]] 的最小复现方法缩小范围。

### 9.5 误把杀进程当成修复

- **现象**：训练卡住或占用资源，于是随意终止进程。
- **原因**：没有先区分程序在计算、等待 I/O、等待 GPU，还是确实失控。
- **判断**：查看日志、CPU/GPU 占用、进程启动命令和 PID。
- **处理**：先 Ctrl+C 或正常停止；确认 PID 后再结束；检查 checkpoint 是否完整。
- **为什么**：终止进程只停止现象，不一定解决根因。

## 10. 与其他笔记的关系

本篇重点负责：

- 终端与 Shell 的基本模型
- cwd、绝对/相对路径和环境变量
- 文件浏览、复制、移动、删除的安全使用
- 管道、重定向、进程和后台训练
- 读取与运行 AI 项目脚本

不在本篇系统展开：

- Conda、Python 解释器与 pip → [[Conda与环境管理]]
- Git 历史、分支和远程同步 → [[Git]]
- traceback、日志设计和系统 Debug → [[Debug与工程基础]]
- Python 语法、函数、模块和类 → [[Python基础]]
- NumPy 数组计算 → [[NumPy]]
- PyTorch 训练、Tensor 和 CUDA API → [[PyTorch]]

## 11. 本篇知识地图

```text
命令行运行 AI 项目
├── 认识 Shell
│   ├── 命令
│   ├── 参数
│   └── 环境变量
├── 确认位置
│   ├── cwd
│   ├── 绝对路径
│   └── 相对路径
├── 操作文件
│   ├── 查看
│   ├── 搜索
│   ├── 复制/移动
│   └── 谨慎删除
├── 连接命令
│   ├── 管道
│   └── 重定向
├── 管理运行任务
│   ├── 进程/PID
│   ├── 日志
│   └── 后台训练
└── 项目启动
    ├── clone
    ├── activate environment
    ├── run script
    └── verify output
```

## 12. 快速查询表

| 我要做什么 | Linux/macOS | PowerShell |
|---|---|---|
| 当前目录 | pwd | Get-Location |
| 列出文件 | ls -la | Get-ChildItem -Force |
| 进入目录 | cd path | Set-Location path |
| 回上一级 | cd .. | Set-Location .. |
| 创建目录 | mkdir -p path | New-Item -ItemType Directory -Force -Path path |
| 复制 | cp source dest | Copy-Item -LiteralPath source -Destination dest |
| 移动/改名 | mv source dest | Move-Item -LiteralPath source -Destination dest |
| 删除文件 | rm file | Remove-Item -LiteralPath file |
| 搜索文件 | rg --files | Get-ChildItem -Recurse -File |
| 查找命令 | which python | Get-Command python |
| 看环境变量 | echo $PATH | $env:PATH |
| 看 Python 路径 | python -c "import sys; print(sys.executable)" | 同左 |
| 看进程 | ps aux | Get-Process |
| 查看日志末尾 | tail -f train.log | Get-Content train.log -Wait |

## 13. 练习题（题目下附参考答案）

### Level 1：概念

1. 终端、Shell、命令和进程分别是什么？
   
   **参考答案：**终端是输入输出界面；Shell 是解释命令的程序；命令是要求程序执行的指令；进程是一个正在运行的程序实例。

2. 什么是当前工作目录？为什么 AI 项目特别容易受它影响？
   
   **参考答案：**当前工作目录是进程启动时所在的位置。相对路径、配置查找、日志和输出目录都可能以它为基准，所以目录错了会造成文件找不到或读错文件。

3. 绝对路径和相对路径有什么区别？
   
   **参考答案：**绝对路径从盘符或根目录开始，位置完整；相对路径相对于当前工作目录，较短但依赖启动位置。

### Level 2：判断

4. 判断：在终端中输入 python，就一定会调用你刚刚激活的 Python 环境。
   
   **参考答案：**错误。要用 sys.executable、which 或 Get-Command 验证实际路径；PATH、Shell 初始化和环境激活都会影响命令解析。

5. 判断：cd 进入项目目录后，Python 环境也会自动切换。
   
   **参考答案：**错误。cd 只改变当前工作目录，不会改变 Conda 环境或 Python 解释器。

6. 判断：把训练放到后台运行后，就不需要查看日志。
   
   **参考答案：**错误。后台只改变进程与终端的关系，仍需通过日志、进程和资源状态确认训练是否正常。

7. 判断：rm 或 Remove-Item 只是把文件从屏幕上隐藏起来。
   
   **参考答案：**错误。删除可能使文件难以恢复，尤其是递归删除目录；必须先确认目标和备份。

### Level 3：读命令

8. 解释下面命令的整体流程：
   
   ```bash
   cd project
   python train.py --config configs/base.yaml
   ```
   
   **参考答案：**先把当前工作目录切换到 project，再调用当前可找到的 Python 执行 train.py，并把 configs/base.yaml 作为 config 选项的值传入。

9. 下面命令为什么有用？
   
   ```bash
   python -c "import sys; print(sys.executable)"
   ```
   
   **参考答案：**它让当前 Python 打印自身的实际文件路径，可用于确认是否运行了预期 Conda 环境中的解释器。

10. 解释管道和重定向的差别。
   
    **参考答案：**管道把前一个命令的输出交给下一个命令继续处理；重定向把输出写入文件或把输入从文件读取，例如 > 保存日志。

### Level 4：应用

11. 你拿到一个远程 AI 仓库，写出从进入目录到运行训练前的检查顺序。
   
    **参考答案：**clone → cd 进入仓库 → pwd/Get-Location 确认位置 → ls/Get-ChildItem 查看文件 → git status/log 确认代码 → 激活环境 → 验证 Python 路径 → 检查配置和数据路径 → 运行入口。

12. 如何把一个长时间训练安全地保存为日志？
   
    **参考答案：**先创建日志目录，再把普通输出和错误输出重定向到日志；Linux 可使用 nohup 和后台运行；之后用 tail -f 或 PowerShell 的 Get-Content -Wait 查看，并记录进程 PID。

13. 你要复制配置文件做实验备份，应该如何降低误覆盖风险？
   
    **参考答案：**先查看源文件和目标文件是否存在，使用明确的源/目标路径复制，使用带日期或实验名的目标文件，并复制后检查目标内容。

### Level 5：Debug

14. 程序提示找不到 data/train.json，但文件明明存在。请给出排查顺序。
   
    **参考答案：**检查当前工作目录 → 解析相对路径对应的绝对路径 → 检查文件实际大小和权限 → 检查配置是否覆盖路径 → 确认运行的是预期仓库副本。不要先重装环境。

15. 终端提示 command not found: conda。你会检查哪些层次？
   
    **参考答案：**检查 Conda 是否安装、当前 Shell 是否初始化、PATH 是否包含 Conda、是否打开了正确终端；Conda 具体排查见 [[Conda与环境管理]]。

16. 后台训练突然消失。如何判断是程序错误、终端断开还是资源问题？
   
    **参考答案：**查看日志最后内容和错误输出，检查进程是否仍在，查看 GPU/CPU/内存资源及系统作业状态，确认启动命令和 nohup/tmux 使用方式，再根据证据判断。

### Level 6：综合科研场景

17. 设计一份 AI 训练启动脚本应包含的最小信息。
   
    **参考答案：**当前目录或项目根目录、Git commit、环境/解释器路径、配置文件、数据根目录、输出目录、随机种子、启动命令和日志位置；关键路径应先检查存在性。

18. 你准备删除一个旧 checkpoint 目录，如何执行得更安全？
   
    **参考答案：**确认目录的绝对路径和归属，列出内容，确认已有备份和不再需要，避免宽泛递归目标；删除后检查目标状态并记录删除原因。

### 综合题

19. Windows 上脚本要求使用 bash，Linux 服务器上却无法直接运行 .ps1 文件。为什么？
   
    **参考答案：**PowerShell 脚本和 Bash 脚本使用不同解释器和语法。应按项目要求选择对应 Shell，或阅读脚本后进行有意识的等价转换，不能只改文件扩展名。

20. 设计“最小可复现命令行排查流程”，用于训练无法启动的问题。
   
    **参考答案：**确认 cwd → 查看仓库状态和 commit → 验证 Python 路径 → 检查入口、配置和数据文件 → 直接运行最小命令并保存完整输出 → 查看日志和进程 → 每次只改一个变量并记录结果。

## 14. 掌握度检查（附答案）

1. 终端和 Shell 有什么区别？
   
   **答案：**终端是交互界面，Shell 是解释和执行命令的程序。

2. 当前工作目录为什么重要？
   
   **答案：**相对路径、配置、日志和输出都可能以它为基准；目录错了，代码可能找不到或读错文件。

3. 如何确认当前 Python 的实际路径？
   
   **答案：**运行 python -c "import sys; print(sys.executable)"，也可以使用 which 或 Get-Command。

4. cd 会不会切换 Conda 环境？
   
   **答案：**不会。cd 只改变目录；环境切换需要 conda activate 等操作。

5. 绝对路径与相对路径什么时候各自更合适？
   
   **答案：**绝对路径明确但可能不便迁移；相对路径便于项目移动但依赖 cwd。科研项目应明确记录路径基准。

6. 管道和重定向有什么区别？
   
   **答案：**管道把输出交给另一个命令处理；重定向把输入输出连接到文件或其他流。

7. 如何查看一个训练进程是否还在运行？
   
   **答案：**Linux 可用 ps、pgrep；PowerShell 可用 Get-Process，并结合日志、PID 和资源状态判断。

8. 为什么不能随意使用递归删除？
   
   **答案：**它可能一次删除大量代码、数据、日志或 checkpoint，通常难以恢复；应先验证明确目标。

9. Shell 脚本对 AI 项目有什么价值？
   
   **答案：**它保存完整启动流程，减少手工错误，使环境、配置、输出和日志路径更容易复现。

10. 训练日志应该记录什么？
    
    **答案：**至少记录启动时间、项目路径、Git commit、Python/环境信息、配置文件、数据路径、关键参数、loss/指标和 checkpoint 位置。

11. 为什么同一条命令在 Windows 和 Linux 上可能不同？
    
    **答案：**Shell、路径分隔符、命令名、权限、换行符和环境变量语法可能不同；不能假设 Bash 和 PowerShell 完全兼容。

12. FileNotFoundError 的第一轮命令行排查是什么？
    
    **答案：**确认 cwd，解析相对路径，检查文件是否存在和可读，检查配置是否覆盖路径，再判断是否是代码问题。

13. 后台运行和可靠任务管理有什么区别？
    
    **答案：**后台运行只是让进程不占用当前交互输入；日志、断线保护、资源调度和失败重启仍需要 tmux、作业系统或项目工具。

14. 为什么命令执行前要先查看状态？
    
    **答案：**可以确认当前目录、目标文件、分支、进程和环境，避免在错误对象上执行高影响操作。

15. 怎样判断一次命令行 Debug 已经完成？
    
    **答案：**问题能稳定复现和验证，根因有证据，修复后原错误消失，相关输出符合预期，并且命令、环境和结论已记录。
