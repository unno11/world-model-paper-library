# Git 与 AI Research Code 版本管理

Git 是 AI 科研工作中经常被低估的一项基础工具。它不只是“把代码上传到 GitHub”，而是帮助你回答：现在运行的是哪一版代码？我改了什么？昨天为什么能跑？这个实验结果对应哪个模型和配置？如果修改失败，如何安全回到上一个状态？

一个真实科研项目通常经历：

```text
clone 仓库
    ↓
阅读 README 和项目结构
    ↓
创建/激活环境，安装依赖
    ↓
运行 baseline
    ↓
修改模型、配置或数据处理
    ↓
比较 diff，保存 commit
    ↓
运行实验并记录 commit
    ↓
切换分支、合并或回退
```

> [!NOTE]
> 本篇聚焦 Git 的科研使用：仓库、工作区、暂存区、commit、branch、remote、clone、pull、push、diff、冲突与实验可追溯性。Conda、Python 环境见 [[Conda与环境管理]]；通用错误排查见 [[Debug与工程基础]]；Shell 命令和路径见 [[Linux与Shell]]；Python 工程结构见 [[Python基础]]。

## 0. 学习目标

学完以后，应该能够：

- 用自己的话解释 repository、working tree、staging area、commit、branch、remote。
- 判断一个目录是不是 Git 仓库，查看当前分支和修改状态。
- 从远程仓库 clone AI 项目，并理解 clone 做了什么、没有做什么。
- 用 status、diff 和 log 检查自己到底改了什么。
- 把有意义的一组修改保存为 commit，并写出可读的 commit message。
- 创建实验分支，避免直接污染稳定代码。
- 区分 fetch、pull、push 和 merge 的作用。
- 处理未跟踪文件、冲突、误改文件和“代码版本对不上”的问题。
- 把 Git commit 与 Conda 环境、配置、数据和 checkpoint 联系起来，建立可复现实验记录。

## 1. 为什么 AI 项目需要 Git？

假设你今天修改了模型：

- 改了一个 Transformer block。
- 把学习率从 1e-4 改成 3e-4。
- 改了数据预处理。
- 修改了 checkpoint 加载逻辑。

训练两天后，结果变好了，但你说不清到底是哪一个修改造成的；或者结果变坏了，却无法恢复到昨天的状态。没有版本管理时，常见做法是复制目录：

```text
project_final/
project_final_v2/
project_final_new/
project_final_really_new/
```

这种方式很快会失控：目录之间有哪些差异不清楚，依赖和数据路径也可能混在一起。

Git 提供了一个更可追踪的模型：

```text
文件发生修改
    ↓
Git 记录差异
    ↓
你选择要保存的修改
    ↓
形成带说明的 commit
    ↓
以后可以比较、切换、合并或回退
```

Git 解决的是“代码和相关文本文件的版本历史”。它不会自动管理 Python 环境、巨大数据集、GPU 驱动或实验结果；这些需要和 Git 配合记录。

## 2. 核心概念

### 2.1 Git

**直觉**：Git 是一个记录文件变化和版本关系的本地工具。

**正式含义**：Git 是分布式版本控制系统。每个 Git 仓库通常包含自己的历史记录，因此你可以在本地查看 commit、比较版本和创建分支，不必每一步都连接服务器。

代码中通常直接写 git，它是命令行程序，不是 GitHub。

### 2.2 Repository（仓库）

**直觉**：仓库是“项目文件 + Git 历史”的整体。

仓库通常包含：

```text
项目源代码
配置文件
README
依赖声明
Git 历史和分支信息
```

.git 是 Git 保存仓库元数据和历史的目录。它不是普通代码目录；不要手动修改或删除它。

GitHub 是托管 Git 仓库的网络平台之一。Git 和 GitHub 不是同一个东西：Git 可以在本地工作，GitHub 是一个远程服务。

### 2.3 Working Tree（工作区）

工作区 / working tree 是你当前在文件系统中看到、编辑和运行的项目文件。

```text
你用编辑器修改 train.py
    ↓
修改首先发生在 working tree
```

工作区的修改不会自动成为 Git 历史。你必须明确查看并保存它们。

### 2.4 Staging Area（暂存区）

暂存区 / staging area 是一个“下一次 commit 准备保存哪些修改”的中间区域。

直觉上可以把它看成装箱台：

```text
工作区：所有当前修改
    ↓ git add
暂存区：你选择下一次保存的修改
    ↓ git commit
历史：正式保存为一个版本
```

暂存区不是远程服务器，也不是最终历史。它允许你把多个文件中的一部分修改组合成一个有明确意义的 commit。

### 2.5 Commit（提交）

commit / 提交是 Git 历史中的一个版本节点，记录一组被保存的文件状态和说明。

直觉：commit 像一个带时间和说明的项目快照，但它不仅仅是复制整个目录；Git 会高效记录内容关系和差异。

一个 commit 通常包含：

- 保存的文件状态。
- 作者和时间。
- commit message。
- 指向父 commit 的关系。
- 一个可用于标识它的 hash，例如 a1b2c3d。

commit 不等于 push。commit 可以只存在本地；push 才是把本地历史发送到远程仓库。

### 2.6 Branch（分支）

branch / 分支是指向某条提交历史的可移动名称。

直觉：从稳定版本分出一条实验路线，在这条路线里修改，不影响主线；实验完成后再合并。

分支不是把整个项目复制一份。它主要是一个指向 commit 的名字，切换时 Git 根据历史把工作区更新到对应状态。

常见分支名：

```text
main / master：主线
experiment/world-model：实验分支
bugfix/data-path：修复分支
```

### 2.7 Remote（远程仓库）

remote / 远程仓库是另一个 Git 仓库的位置，通常是 GitHub、GitLab 或服务器上的仓库。

远程仓库的 URL 可以是 HTTPS 或 SSH。origin 通常只是默认远程名称，不是特殊的服务器。

```text
本地仓库  ← fetch/pull ← 远程仓库
本地仓库  → push       → 远程仓库
```

### 2.8 HEAD

HEAD 表示当前正在查看和工作的 commit，通常指向当前分支。

例如：

```text
HEAD → main → 某个最新 commit
```

如果 HEAD 没有指向分支而是直接指向某个 commit，叫 detached HEAD（分离头指针）。这适合查看历史版本，但在上面创建的 commit 可能不容易从分支入口找到，所以实验前要先理解当前状态。

## 3. Git 的整体工作流

```text
远程仓库
    ↓ git clone
本地仓库 + 工作区
    ↓ 编辑/运行
工作区发生修改
    ↓ git diff / git status
查看差异和状态
    ↓ git add
选择进入暂存区的修改
    ↓ git diff --staged
检查下一次提交内容
    ↓ git commit
保存本地历史
    ↓ git log
确认版本节点
    ↓ git push
发送到远程仓库
```

最重要的原则：每一步都可以检查，不要把 add、commit、push 当成一个不可观察的动作。

## 4. 从远程仓库开始：clone

### 4.1 git clone 做什么？

```bash
git clone https://github.com/example/project.git
```

逐段理解：

| 部分 | 含义 |
|---|---|
| git | 调用 Git 程序 |
| clone | 复制远程仓库到本地，并建立 Git 历史和远程关联 |
| URL | 远程仓库地址 |
| 末尾可选目录名 | 指定本地保存目录，不写时通常根据仓库名生成 |

clone 通常会：

- 下载代码和 Git 历史的一部分或全部。
- 创建本地项目目录。
- 创建 .git 元数据目录。
- 设置名为 origin 的远程仓库。
- 检出默认分支到工作区。

clone 不会自动：

- 创建或激活 Conda 环境。
- 安装 Python 依赖。
- 下载数据集或 checkpoint。
- 保证 CUDA 可用。
- 自动运行训练。

因此科研项目常见的启动顺序是：

```bash
git clone <repository-url>
cd <repository-directory>
git status
conda create -n project python=3.10
conda activate project
python -m pip install -r requirements.txt
```

环境细节见 [[Conda与环境管理]]。

### 4.2 clone 后的第一轮检查

```bash
cd <repository-directory>
git status
git remote -v
git branch --show-current
git log --oneline -5
```

观察：

- git status 是否显示工作区干净。
- git remote -v 是否是你预期的远程地址。
- 当前分支叫什么。
- 最近的 commit 是什么，代码处于哪一个版本。

如果仓库不是你自己的，先不要直接 push；先阅读 README、许可证和贡献说明。

## 5. status、diff 与 log

### 5.1 git status

```bash
git status
```

它回答：

- 当前在哪个分支？
- 哪些文件被修改？
- 哪些文件已暂存？
- 哪些文件是 Git 尚未跟踪的新文件？
- 当前分支是否领先或落后远程分支？

status 是最值得频繁运行的 Git 命令。修改前、修改后、提交前、切换分支前都应看一次。

### 5.2 git diff

```bash
git diff
```

查看工作区中尚未暂存的修改。

```bash
git diff --staged
```

查看暂存区中、下一次 commit 将要保存的修改。

两者区别非常重要：

```text
git diff          ：我改了什么，但还没有 add？
git diff --staged ：我 add 了什么，下一次 commit 会保存什么？
```

### 5.3 git log

```bash
git log --oneline --decorate --graph -10
```

查看最近 10 个 commit，并用简短图形显示分支关系。

单个 commit：

```bash
git show <commit-id>
```

它可以查看该 commit 的说明、元信息和差异。

### 5.4 为什么不能只看文件当前内容？

当前文件只告诉你“现在是什么”，Git 历史还可以告诉你：

- 什么时候变成这样。
- 谁做了修改。
- 这次修改解决了什么问题。
- 修改前是什么。
- 哪个 commit 可能引入了 Bug。

这对 Debug 很重要：如果训练昨天正常、今天失败，可以比较两个 commit，而不是凭记忆猜测。见 [[Debug与工程基础]]。

## 6. 保存修改：add 与 commit

### 6.1 git add

```bash
git add train.py
git add configs/base.yaml
```

git add 把指定文件当前的修改放入暂存区。它不是“保存到远程”，也不是“创建 commit”。

查看状态和暂存差异：

```bash
git status
git diff --staged
```

可以一次添加目录，但要谨慎：

```bash
git add src/
```

不要不加思考地运行 git add .，尤其是在训练目录中可能有 checkpoint、日志、数据副本或密钥文件时。先看 git status 和 .gitignore。

### 6.2 git commit

```bash
git commit -m "Fix dataset path resolution"
```

逐段解释：

| 部分 | 含义 |
|---|---|
| git | 调用 Git |
| commit | 把暂存区内容保存为本地历史节点 |
| -m | 直接提供 commit message |
| 引号内容 | 说明这一组修改做了什么 |

commit 只保存暂存区，不会自动保存所有工作区修改。因此提交前应该：

```bash
git status
git diff
git diff --staged
git commit -m "..."
git status
```

### 6.3 好的 commit message

好的 message 让未来的你读历史时知道发生了什么：

```text
Fix relative dataset path in training entrypoint
Add shape validation for observation batch
Record baseline configuration for JEPA experiment
```

避免：

```text
update
fix
final
test
aaa
```

一个 commit 最好表达一个相对完整的意图。不要把“修改模型、删除数据、重命名脚本、升级依赖”混成一个无法解释的巨型 commit。

### 6.4 commit 前后的验证

提交后：

```bash
git status
git log -1 --oneline
git show --stat --oneline HEAD
```

理想状态是：工作区干净，最近 commit 的文件列表和说明符合预期。commit 不是验证代码正确；仍需要运行测试或最小实验。

## 7. 分支：安全地进行实验

### 7.1 为什么使用分支？

如果你直接在 main 上试验大型模型改动，可能让原本可以运行的 baseline 变得不可用。分支允许你保留一条稳定基线，同时创建实验路线。

### 7.2 创建并切换分支

现代 Git 推荐：

```bash
git switch -c experiment/latent-planning
```

含义：创建名为 experiment/latent-planning 的新分支，并立即切换过去。

查看分支：

```bash
git branch
git branch --show-current
```

切回主分支：

```bash
git switch main
```

旧项目中也可能看到：

```bash
git checkout -b experiment/latent-planning
```

它能完成类似工作，但 switch 更明确地表达“切换分支”。

### 7.3 分支命名与实验记录

分支名应体现目的：

```text
experiment/vision-pretraining
experiment/robot-policy-ablation
bugfix/checkpoint-path
```

分支不是实验记录的全部。还要记录：配置、数据版本、环境、commit、随机种子和启动命令。

## 8. 从远程同步：fetch、pull、push

### 8.1 git fetch

```bash
git fetch origin
```

从远程获取新的 commit 和引用信息，但通常不直接修改当前工作区文件。

直觉：先把远程的“最新消息”拿到本地，再决定怎么整合。

### 8.2 git pull

```bash
git pull origin main
```

通常可以理解为：先 fetch，再把远程更新整合到当前分支（具体整合方式可能是 merge 或 rebase，取决于配置和命令参数）。

pull 可能修改你的工作区，因此在有未保存修改时先运行 git status。不要在不理解当前状态时盲目 pull。

### 8.3 git push

```bash
git push origin experiment/latent-planning
```

把本地分支的 commit 发送到远程仓库。

第一次推送新分支时常见：

```bash
git push -u origin experiment/latent-planning
```

-u 建立本地分支与远程跟踪分支的关联，以后可以更简洁地使用 git push 和 git pull。

push 不会自动上传未 commit 的修改。只有已经进入本地 commit 历史的内容，才有机会被 push。

### 8.4 三者对比

| 命令 | 主要动作 | 是否通常改当前工作区 |
|---|---|---|
| fetch | 获取远程历史信息 | 通常不改 |
| pull | 获取并整合远程更新 | 可能会改 |
| push | 把本地 commit 发送到远程 | 不改本地工作区 |

## 9. .gitignore 与科研文件

.gitignore 用于声明 Git 不应默认跟踪的文件模式。

AI 项目通常考虑忽略：

```text
__pycache__/
*.pyc
.pytest_cache/
logs/
outputs/
checkpoints/
*.ckpt
*.pt
*.pth
数据目录或本地缓存目录
.env
```

但不能机械复制列表。是否忽略 checkpoint、配置和日志取决于项目需求：

- 巨大模型权重通常放在专门的模型存储，而不是普通 Git 历史。
- 复现实验所需的小型配置应该提交。
- API key、密码、token、私有路径绝不能提交。
- 已经被 Git 跟踪的文件，即使后来加入 .gitignore，也不会自动停止跟踪，需要单独处理。

> [!WARNING]
> 一旦密钥被 commit 并 push，不能只靠删除文件解决；历史中可能仍有泄露。应立即撤销/轮换密钥，并按平台流程处理历史清理。

## 10. 合并与冲突

### 10.1 merge 的直觉

假设 main 和实验分支分别产生了 commit：

```text
main:       A ── B ── C
                 \
experiment:       D ── E
```

把实验分支合回 main：

```bash
git switch main
git merge experiment/latent-planning
```

Git 会尝试自动整合两个分支的修改。

### 10.2 什么是冲突？

当 Git 无法自动决定同一处代码应保留哪一种修改，就产生 merge conflict（合并冲突）。文件中可能出现：

```text
<<<<<<< HEAD
main 分支内容
=======
实验分支内容
>>>>>>> experiment/latent-planning
```

处理流程：

1. 运行 git status，查看冲突文件。
2. 打开文件，理解两边修改的意图。
3. 手动编辑成最终正确版本，删除冲突标记。
4. 运行语法检查、测试或最小实验。
5. git add <resolved-file>，表示已解决该文件。
6. 完成 merge commit，或按当前操作流程继续。

解决冲突不是简单地“全部选当前”或“全部选传入”。要结合模型、配置和实验意图判断。

### 10.3 取消一次未完成的 merge

如果发现自己还没有能力安全解决：

```bash
git merge --abort
```

它尝试回到 merge 开始前的状态。执行前先看 git status；不要把它当作删除所有工作的方法。

## 11. 撤销与回退：先区分状态

“撤销”不是一个单一动作。先判断修改在哪个区域。

### 11.1 取消暂存，但保留工作区修改

```bash
git restore --staged train.py
```

它表示：不让这份修改进入下一次 commit，但文件修改仍保留在工作区。

### 11.2 丢弃工作区修改

```bash
git restore train.py
```

这会丢弃该文件未提交的工作区修改，具有破坏性。确认目标文件和内容确实可以丢弃后再执行。不要对整个仓库或宽泛通配符随意使用。

### 11.3 生成一个反向 commit：git revert

```bash
git revert <commit-id>
```

它创建一个新的 commit，反向抵消指定 commit 的效果。对于已经共享到远程的历史，revert 通常比重写公共历史更安全，因为原来的历史仍然存在。

### 11.4 reset 目前知道即可

git reset 可以移动分支指针并改变暂存区或工作区，功能强大也容易丢失工作。当前阶段先记住：

- 共享历史回退优先了解 revert。
- 不要在没有确认状态和备份的情况下使用强制 reset。
- 任何破坏性命令前先运行 git status、git log 和必要的 diff。

## 12. AI Research Repository 的完整工作流

### 12.1 第一次运行 baseline

```bash
git clone <repository-url>
cd <repository-directory>
git status
git log -1 --oneline
conda create -n world-model python=3.10
conda activate world-model
python -m pip install -r requirements.txt
python train.py --config configs/baseline.yaml
```

Git 在这里负责：

- 获取代码。
- 告诉你当前 commit。
- 保存后续代码修改。
- 帮助恢复 baseline。

Conda 负责环境；数据下载、GPU 分配、训练输出和 checkpoint 需要另外处理。见 [[Conda与环境管理]]。

### 12.2 开始一个实验

```bash
git switch -c experiment/ablation-context-length
# 修改配置或代码
git status
git diff
python train.py --config configs/ablation.yaml
git add configs/ablation.yaml src/
git diff --staged
git commit -m "Run context length ablation"
git log -1 --oneline
```

实验记录至少包含：

```text
代码 commit
Conda 环境/关键包版本
配置文件
数据版本和路径
随机种子
GPU 与驱动信息
启动命令
输出指标和 checkpoint 路径
```

这样你才能回答“这个结果对应哪一版代码”。

### 12.3 代码版本与 checkpoint 的关系

checkpoint 通常包含模型参数，有时还包含优化器状态、epoch 和配置，但不一定包含完整源代码和环境。一个 checkpoint 的可用性依赖：

- 模型定义与保存时兼容。
- 配置参数一致或经过明确转换。
- 数据预处理一致。
- 依赖和设备支持相关操作。

因此不要只保存 model.pth；至少同时保存 commit、配置和环境信息。

## 13. 常见错误与 Debug

### 13.1 not a git repository

- **现象**：运行 Git 命令时提示当前目录不是 Git 仓库。
- **原因**：没有进入 clone 得到的项目目录，或 .git 不存在。
- **判断**：运行 Get-Location/pwd，检查目录中是否有 .git。
- **处理**：进入正确仓库目录，或确认 clone 是否成功。
- **为什么**：Git 命令通常从当前目录向上查找仓库边界；项目目录和仓库目录需要分清。

### 13.2 nothing to commit, working tree clean

- **现象**：你以为改了代码，但 Git 说没有修改。
- **原因**：文件没保存、改的是另一个副本、修改被忽略、当前分支不是预期分支，或内容最后恢复原样。
- **判断**：检查编辑器文件路径、git branch --show-current、git status --ignored 和 git diff。
- **处理**：确认运行和编辑的是同一个仓库副本；不要直接删除 .git 或重新 clone 来掩盖问题。

### 13.3 git add . 加入了不该提交的文件

- **现象**：暂存区出现日志、checkpoint、数据或密钥。
- **原因**：添加范围过大，或 .gitignore 未配置。
- **判断**：运行 git status 和 git diff --staged，逐个查看。
- **处理**：对不应提交的文件执行 git restore --staged <file>；补充 .gitignore；如果密钥已经提交，立即轮换。

### 13.4 Your branch is behind 或 push 被拒绝

- **现象**：远程有本地没有的 commit，push 被拒绝。
- **原因**：他人或另一台机器已经更新了远程分支。
- **判断**：先 git fetch origin，再查看分支状态和日志。
- **处理**：理解远程更新后，选择 merge 或 rebase 整合，再运行测试，最后 push。不要一看到拒绝就强制 push。

### 13.5 merge conflict

- **现象**：Git 标记冲突文件，无法自动完成合并。
- **原因**：两个分支修改了重叠区域，Git 无法判断最终内容。
- **判断**：git status 查看冲突文件，阅读冲突标记两侧的代码和提交意图。
- **处理**：手动合并 → 删除标记 → 测试/最小运行 → git add → 完成合并；不确定时 git merge --abort。

### 13.6 Detached HEAD

- **现象**：Git 提示处于 detached HEAD 状态。
- **原因**：直接检出了某个 commit 或远程引用，而不是分支。
- **判断**：运行 git status 和 git branch --show-current。
- **处理**：只是查看时可以保持；要继续实验，创建分支：git switch -c experiment/from-old-commit。
- **为什么**：否则新 commit 可能没有稳定的分支名称指向它。

### 13.7 大文件或密钥误提交

- **现象**：push 失败、仓库过大，或发现 token 已进入历史。
- **原因**：把 checkpoint、数据或秘密文件加入了 Git。
- **判断**：检查 git status、commit 内容和远程平台提示。
- **处理**：未 commit 时取消暂存；已 commit 但未 push 时谨慎修正；已 push 的密钥立即撤销/轮换，并按仓库管理员流程清理历史。

## 14. 与其他笔记的关系

本篇重点负责：

- Git 仓库和版本历史
- clone、status、diff、add、commit、branch
- fetch、pull、push、merge 与冲突
- AI 科研代码的版本追踪和实验可复现

以下内容只建立链接，不在本篇系统展开：

- Python 环境、解释器、pip 与依赖 → [[Conda与环境管理]]
- traceback、日志、断点和通用排错 → [[Debug与工程基础]]
- Shell、路径和命令行语法 → [[Linux与Shell]]
- Python 语法、模块和工程结构 → [[Python基础]]
- 数据数组 → [[NumPy]]
- Tensor、模型、训练与 checkpoint 内部结构 → [[PyTorch]]

本篇不会深入 Git 内部对象存储、复杂 rebase 策略、CI/CD、Git LFS 和大型团队权限管理；当前阶段知道它们存在即可。

## 15. 本篇知识地图

```text
AI 项目版本可追溯
├── 获取
│   └── clone → 本地仓库 + origin
├── 观察
│   ├── status
│   ├── diff
│   └── log/show
├── 保存
│   ├── add：工作区 → 暂存区
│   └── commit：暂存区 → 本地历史
├── 实验隔离
│   └── branch / switch
├── 协作同步
│   ├── fetch
│   ├── pull
│   └── push
├── 整合
│   └── merge → 处理冲突 → 验证
└── 复现科研结果
    ├── commit
    ├── environment
    ├── config/data
    ├── checkpoint
    └── command/metrics
```

## 16. 快速查询表

| 我要做什么 | 命令 | 作用 |
|---|---|---|
| 查看 Git 版本 | git --version | 确认 Git 可用 |
| 克隆仓库 | git clone <url> | 获取远程项目和历史 |
| 查看状态 | git status | 查看分支、修改、暂存和同步状态 |
| 查看当前分支 | git branch --show-current | 确认正在工作的分支 |
| 查看远程 | git remote -v | 确认远程地址 |
| 查看工作区差异 | git diff | 查看未暂存修改 |
| 暂存文件 | git add <file> | 选择下一次提交内容 |
| 查看暂存差异 | git diff --staged | 检查即将提交的内容 |
| 创建提交 | git commit -m "message" | 保存本地版本节点 |
| 查看历史 | git log --oneline -10 | 查看最近提交 |
| 创建实验分支 | git switch -c experiment/name | 创建并切换分支 |
| 获取远程更新 | git fetch origin | 获取远程历史，不直接整合 |
| 推送分支 | git push -u origin branch | 上传本地 commit |
| 查看某次提交 | git show <commit> | 查看提交内容 |
| 取消合并 | git merge --abort | 尝试回到合并前状态 |

## 17. 练习题（题目下附参考答案）

### Level 1：概念

1. Git 和 GitHub 是什么关系？

   **参考答案：**Git 是本地可运行的版本控制工具；GitHub 是托管 Git 仓库的网络平台之一。没有 GitHub，Git 也可以在本地记录历史。

2. repository、working tree、staging area、commit 分别是什么？

   **参考答案：**repository 是项目文件和 Git 历史的整体；working tree 是当前编辑的文件；staging area 是下一次提交的候选内容；commit 是正式保存到本地历史的版本节点。

3. 为什么 commit 不等于 push？

   **参考答案：**commit 把暂存区内容保存到本地历史；push 才把本地 commit 发送到远程仓库。可以 commit 但暂时不 push。

### Level 2：判断

4. 判断：git clone 后，项目依赖和数据集已经准备好了。

   **参考答案：**错误。clone 通常只取得代码、Git 历史和远程关联，不会自动创建 Conda 环境、安装依赖或下载数据。

5. 判断：运行 git add . 就等于把所有文件提交到 GitHub。

   **参考答案：**错误。git add . 只是把部分工作区内容放入暂存区；之后还要检查、commit，最后才可能 push 到远程。

6. 判断：分支就是完整复制一份项目目录。

   **参考答案：**错误。分支主要是指向 commit 历史的名称；切换分支时 Git 根据历史更新工作区，但它不是普通文件夹复制。

7. 判断：程序能运行就说明当前 commit、环境和实验结果已经可复现。

   **参考答案：**错误。还需要记录依赖、配置、数据、硬件、随机种子、启动命令和输出；Git 只解决代码版本的一部分。

### Level 3：读命令

8. 逐段解释 git clone https://github.com/example/project.git。

   **参考答案：**调用 Git；clone 表示复制远程仓库到本地并建立历史与远程关联；URL 指定远程仓库地址。它不会安装 Python 依赖或运行训练。

9. git diff 与 git diff --staged 有什么区别？

   **参考答案：**前者查看尚未暂存的工作区修改；后者查看暂存区中下一次 commit 将要保存的修改。

10. 为什么提交前要同时运行 git status、git diff 和 git diff --staged？

    **参考答案：**status 看整体状态；diff 看还有哪些修改未加入提交；diff --staged 确认真正会进入 commit 的内容，避免漏提交或误提交。

11. 解释 git switch -c experiment/latent-planning。

    **参考答案：**创建一个名为 experiment/latent-planning 的新分支，并立即切换到该分支。

### Level 4：应用

12. 从 clone 一个 AI 仓库到运行第一次 baseline，写出合理顺序。

    **参考答案：**git clone → cd 进入仓库 → git status 和 git log 确认版本 → 创建并激活 Conda 环境 → 安装依赖 → 准备数据/配置 → 运行 baseline → 记录 commit、环境和命令。

13. 你要修改数据预处理，但不想破坏主线。应该怎么做？

    **参考答案：**先确认工作区干净，使用 git switch -c experiment/data-preprocess 创建实验分支，在分支中修改、验证并提交；主线保持可运行。

14. 如何把一次实验保存成可追踪状态？

    **参考答案：**提交代码和配置的 commit，记录 Conda/包版本、数据版本与路径、随机种子、GPU 信息、完整启动命令、指标和 checkpoint 路径。

### Level 5：Debug 与协作

15. 运行 git status 显示 nothing to commit, working tree clean，但你确定改过代码。排查什么？

    **参考答案：**检查文件是否保存、编辑器打开的路径是否是当前仓库、当前分支是否正确、修改是否被恢复或忽略；可检查 Get-Location、文件绝对路径、git branch --show-current 和 git status --ignored。

16. git push 被拒绝并提示远程分支有本地没有的提交。应该怎么做？

    **参考答案：**先 git fetch origin 查看远程更新和当前状态，理解差异后用 merge 或 rebase 整合，解决冲突并测试，再 push。不要未经判断直接强制 push。

17. 合并冲突出现后，正确的处理顺序是什么？

    **参考答案：**git status 找出冲突文件 → 阅读两边修改意图 → 手动编辑为正确版本并删除冲突标记 → 运行测试/最小实验 → git add 标记已解决 → 完成合并；不确定时可 git merge --abort。

18. 一个 token 被 commit 但还没有 push，和已经 push，处理上有什么不同？

    **参考答案：**未 push 时可以在本地谨慎修正历史并确保 token 不再保存；已 push 时应立即撤销/轮换 token，因为远程历史或缓存可能已经暴露，再按平台流程清理历史。

### Level 6：综合科研场景

19. 你发现今天的训练结果与昨天不同。如何利用 Git 缩小范围？

    **参考答案：**比较两次运行的 commit、git log 和 git diff，确认配置、依赖、数据和启动命令是否同时变化；必要时检出昨天的 commit 做最小复现实验，再逐项引入修改。

20. 设计一个安全的实验分支工作流，要求主线可恢复、实验可复现、误改可发现。

    **参考答案：**从干净主线创建实验分支；每次只做一个有明确意图的修改；运行 status/diff 检查；用 .gitignore 排除数据、日志和秘密；提交配置与代码；记录环境、数据、随机种子、命令和 checkpoint；定期运行最小验证；合并前处理冲突并测试，必要时保留可回退 commit。

## 18. 掌握度检查（附答案）

学完本篇后，先尝试独立回答每个问题，再展开对应答案。

1. Git 主要解决什么问题？

   **答案：**记录代码和相关文本文件的版本历史，让修改可比较、可追踪、可回退和可协作。

2. Git 和 GitHub 的区别是什么？

   **答案：**Git 是版本控制工具；GitHub 是托管远程 Git 仓库的平台之一。

3. working tree、staging area、commit 的关系是什么？

   **答案：**工作区是当前修改；git add 选择修改进入暂存区；git commit 把暂存区内容保存为本地历史节点。

4. git status 为什么应该频繁运行？

   **答案：**它能显示当前分支、未跟踪文件、未暂存/已暂存修改以及与远程的基本状态，是判断下一步动作的总览。

5. git clone 做了什么，没做什么？

   **答案：**它获取代码和 Git 历史、建立本地仓库和远程关联；它不会创建 Python 环境、安装依赖、准备数据或运行训练。

6. git diff 与 git diff --staged 分别回答什么问题？

   **答案：**前者回答“工作区还有什么修改没暂存”；后者回答“暂存区中下一次 commit 会保存什么”。

7. 什么是有意义的 commit？

   **答案：**它包含一组相对完整、可解释、可验证的修改，并用 message 说明目的，而不是把大量无关变化混成一个提交。

8. 为什么科研实验最好使用分支？

   **答案：**分支可以隔离实验改动，保留可运行的主线 baseline，并让不同实验路线可以比较、合并或放弃。

9. fetch、pull、push 有什么区别？

   **答案：**fetch 获取远程历史信息；pull 获取并整合远程更新；push 把本地 commit 发送到远程。

10. Git 冲突是什么？

    **答案：**两个分支修改了重叠内容，Git 无法自动判断最终版本时产生冲突，需要人理解两边意图并手动整合。

11. 为什么不能随便执行 git add .？

    **答案：**它可能把日志、checkpoint、数据、缓存或密钥加入暂存区；提交前必须检查 status、.gitignore 和 diff --staged。

12. 已经 push 的秘密文件为什么不能只删除当前文件？

    **答案：**秘密可能仍存在 Git 历史、远程缓存或他人 clone 中；应立即撤销/轮换秘密，并按流程清理历史。

13. detached HEAD 表示什么？实验时怎么办？

    **答案：**当前 HEAD 直接指向某个 commit 而不是分支；如果要在此基础上继续实验，应创建新分支，例如 git switch -c experiment/from-old-commit。

14. 如何判断一个 AI 实验对应哪一版代码？

    **答案：**记录实验启动时的 commit hash/分支，同时保存配置、环境、数据版本、随机种子、启动命令和 checkpoint 信息。

15. 什么时候应该使用 revert，为什么要谨慎使用破坏性回退？

    **答案：**共享历史需要撤销某次修改时通常可用 revert，因为它创建新的反向 commit；强制 reset 等操作可能丢失或重写他人依赖的历史，必须先确认状态和影响。
