---
type: literature-note
project: jepa-wm
status: 已完成首轮精读
last_updated: 2026-07-22
source: https://arxiv.org/pdf/2512.24497
tags:
  - 文献/JEPA-WM
  - 文献/世界模型
  - 项目/jepa-wm
---

# JEPA-WM 论文精读：What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

> [!info] 论文信息
> Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, Yann LeCun；Meta FAIR / INRIA / ENS / NYU。arXiv:2512.24497v2，2026-01-08。论文研究的不是一个全新单一算法，而是对 JEPA-WM 这一类“在预训练视觉表征空间中学习动作条件动力学并进行规划”的系统性设计选择研究。

论文链接：[arXiv PDF](https://arxiv.org/pdf/2512.24497) · [官方代码仓库](https://github.com/facebookresearch/jepa-wms)
![Pasted image 20260726140509.png](../../%E5%9B%BE%E7%89%87%E5%BC%95%E7%94%A8/a52e1974f0-Pasted%20image%2020260726140509.png)
## 1. 一句话理解

JEPA-WM 的核心不是生成未来像素，而是学习一个 predictor：给定当前观测的 latent、历史上下文和动作序列，预测未来观测的 latent；规划时在 latent 空间展开候选动作，用预测 latent 与目标 latent 的距离评价动作，并通过 CEM 等优化器反复改进动作分布。

![JEPA-WM框架图.svg](../../%E5%9B%BE%E7%89%87%E5%BC%95%E7%94%A8/719f38b7fb-JEPA-WM%E6%A1%86%E6%9E%B6%E5%9B%BE.svg)

> [!note] 图示说明
> 上图是依据论文 Figure 1 重绘的项目内 SVG：左侧是训练，右侧是规划。论文原图特别说明，action encoder 和 proprioceptive encoder 为了可读性没有显式画出；本图把它们补充出来，便于对照代码。

## 2. 论文解决什么问题

论文将 JEPA-WM 形式化为一类统一 recipe，而不是宣称提出一个全新的 world-model 算法。研究问题是：在冻结的预训练视觉 encoder 上，哪些动力学 predictor、训练目标和 planner 选择最有效，尤其是在 navigation 和 manipulation 中？

作者系统研究：

- planner：CEM、Nevergrad/NG、Adam、GD，以及 L1/L2 latent distance；
- training：teacher forcing、multistep rollout loss、训练 context `W`；
- inputs：是否加入 proprioception；
- encoder：DINOv2、DINOv3、V-JEPA、V-JEPA-2；
- predictor：feature conditioning、sequence conditioning、AdaLN、RoPE/sincos；
- scale：encoder size、predictor width 与 depth；
- datasets：Wall、PointMaze/Maze、Push-T、Metaworld、DROID、RoboCasa。

## 3. 统一模型定义

### 3.1 状态编码

全局状态 encoder 为：

```text
E_{φ,θ} = (E^vis_φ, E^prop_θ)
```

- `E^vis_φ`：预训练视觉 encoder，训练 JEPA-WM predictor 时冻结；
- `E^prop_θ`：可选 proprioceptive encoder，与 predictor、action encoder 一起训练；
- `A_θ`：动作 encoder。

因此，论文研究重点在“如何学习 dynamics predictor”，而不是重新学习通用视觉表示。

### 3.2 训练目标

基本 teacher-forcing 目标是：

```text
z_t = E(o_t)
ẑ_{t+1} = P(z_t, a_t)
L_1 = distance(ẑ_{t+1}, E(o_{t+1}))
```

多步 rollout 训练增加：

```text
L = L_1 + L_2 + ... + L_k
```

其中后续预测使用模型自己的前一步预测，而不是始终喂 ground-truth latent；论文采用 truncated BPTT，并在多步训练中研究最多 6-step loss。

### 3.3 规划目标

从初始观测 `o_t` 开始，对候选动作序列做 predictor unroll，得到 `G(o_t, a_{t:t+H-1})`。若使用 proprioception，规划代价为：

```text
L_p^α = L_vis(predicted_state, goal_state)
      + α L_prop(predicted_proprio, goal_proprio)
```

论文最终模型在 simulation 中通常使用 `α=0.1`；DROID/Robocasa 为了和 V-JEPA-2-AC 比较使用 `α=0`。

## 4. 关键实验发现

### 4.1 planner 是结果的一等公民

- CEM L2 在整体上表现最好，因此后续设计选择实验固定为 CEM L2；
- Adam/GD 在 Metaworld 这类 cost landscape 较平滑、目标较贪心可达的任务上表现好；
- 在 Wall、Maze、Push-T 等需要绕行的 2D navigation 中，梯度规划容易陷入局部最优；
- NG 在 DROID/Robocasa 上接近 CEM，但需要较少 planner 超参数调节，是迁移到新任务时的实用替代方案；
- 全部 planner 设置中 L2 distance 持续优于 L1。

### 4.2 rollout loss 存在最佳点，不是越长越好

simulation 中从 teacher forcing 增加到 2-step rollout 通常提升成功率，但继续增加 rollout steps 可能下降。论文的解释是：规划时最大 prediction context 为 `Wp=2`，过长的训练 rollout 可能使模型偏离测试时真正使用的预测任务；DROID 这类真实机器人数据反而偏好 6-step rollout。

### 4.3 proprioception 有稳定帮助

加入 proprioception 的模型总体优于纯视觉模型。在 Metaworld 中，它能提供更精确的末端位置/距离，减少抵达目标后振荡；在 2D navigation 中，也有助于生成更精确的动作计划。

### 4.4 context 的关键作用是推断速度

`W=1` 到 `W=2` 有明显性能差距，说明 predictor 至少需要两帧上下文推断速度；三帧可帮助推断加速度。过长 context 会减少可切出的独立轨迹片段，使固定计算预算下的训练信号变少。必须满足：

```text
W_p <= W
```

否则规划时要求 predictor 使用训练阶段没有见过的 context 长度，unroll 会快速退化。

### 4.5 encoder 的局部空间能力比“视频”标签更重要

DINO 系列优于 V-JEPA 系列，作者推断主要原因是 DINO 的细粒度目标分割/局部特征更适合精确定位 agent 和物体。DINOv3 的优势主要出现在更写实的 DROID/Robocasa；在 Maze/Wall 上，DINOv3 反而收敛更慢且最终成功率更低。

### 4.6 AdaLN 是平均意义上的强选择，但不是所有任务最优

AdaLN + RoPE 的跨环境平均表现最好，可能因为动作信息在 predictor 的每个 block 中注入，减少深层传播时的动作信息衰减；但 Metaworld 上 sincos + feature conditioning 可能更好，说明 conditioning 方案具有任务依赖性。

### 4.7 规模扩展依赖任务复杂度

在 simulation 中扩大 encoder/predictor 宽度或 predictor depth 没有明显收益；在 DROID/Robocasa 中，更大的 encoder 和更深的 predictor 带来一致提升。不能把“大模型一定更好”作为 JEPA-WM 的一般结论。

## 5. 论文最终模型与结果

Simulation 的最终组合：

```text
视觉 encoder：DINOv2 ViT-S
predictor：ViT-S，depth=6，AdaLN conditioning，RoPE
输入：视觉 + proprioception
训练：2-step rollout loss，最大训练 context W=3
规划：CEM L2
```

DROID/Robocasa 的最终组合使用 DINOv3 ViT-L、ViT-L predictor depth 12、无 proprioception；所有环境均采用 CEM L2。

| Model | Maze | Wall | Push-T | MW-Reach | MW-Reach-Wall |
|---|---:|---:|---:|---:|---:|
| DINO-WM | 81.6 ± 3.4 | 64.1 ± 4.6 | 66.0 ± 4.7 | 44.8 ± 8.9 | 35.1 ± 9.4 |
| Ours | **83.9 ± 2.3** | **78.8 ± 3.9** | **70.2 ± 2.8** | **58.2 ± 9.3** | **41.6 ± 10.0** |

论文最终模型每个模型使用 3 个 seeds；每个 epoch 通常评估 96 个独立 episode，DROID 使用 64、Robocasa 使用 32；最终图表对最后 10 个 epoch（DROID 为 100 个 epoch）的成功率做聚合。因此你的 released checkpoint 单点复现结果不能直接和表中均值当作同一统计口径。

## 6. 对当前复现工作的直接指导

### 已对齐的部分

- 你当前 Wall、PointMaze、Push-T、MW-Reach-Wall 的 released checkpoint 结果与论文表格一致到合理统计范围内；
- 你当前使用的 simulation 方向应以 `CEM L2` 为主评估协议；
- `W=3`、2-step rollout、proprioception 和 AdaLN/RoPE 是优先核对的配置项；
- 多 seed、多 episode 和最后 epoch 聚合解释了单 checkpoint 与论文均值的差异。

### Stage 2 训练时必须锁定

- 训练最大 context `W` 与 planning context `Wp` 的关系，不能出现 `Wp > W`；
- rollout loss 的定义，确认后续 rollout 使用 prediction 而不是 ground-truth latent；
- proprioceptive loss 是否参与 predictor/action/proprio encoder 的联合训练；
- predictor conditioning、depth、encoder 类型和输入分辨率；
- CEM 的 horizon、candidate 数、iteration 数、step 数、L1/L2 metric；
- 每个 seed 的完整 YAML、commit、日志、checkpoint hash、episode-level 原始指标。

## 7. 我的判断

这篇论文最重要的贡献不是提出一个复杂的新模块，而是证明 JEPA-WM 的成功高度依赖“表征、训练展开方式、context、planner”之间的匹配。对复现而言，planner 与统计协议和网络结构同等重要；只复现 predictor loss 而没有锁定 CEM/L2、`W/Wp`、episode 数和多 seed 聚合，不能称为完整复现。

对你后续想研究的部分可观测/多假设世界模型，论文提供的 baseline 仍然是确定性单轨迹 predictor。合理的扩展顺序是：先复现确定性历史 context 的上限，再比较 ensemble/多假设 rollout 的不确定性是否改善规划；不要在 baseline 尚未完成多 seed training reproduction 前直接引入复杂 belief model。

## 8. 待继续精读

- Appendix B：训练细节、loss 变体和 predictor 配置；
- Appendix C：Wall、Maze、Push-T、Metaworld 的数据与 goal 采样；
- Appendix D：CEM/NG/Adam/GD 的精确超参数；
- Appendix E：unroll metrics 与 success rate 的相关性，尤其是“预测准确不必然带来规划成功”的证据。

## 关联笔记

- JEPA-WM-研究路线图
- JEPA-WM-阶段进度
- JEPA-WM-Stage1结果与证据
- JEPA-WM-服务器证据采集Prompt
