# Diffusion Policy：用生成过程建模多种机器人动作

Diffusion Policy 把动作序列看成需要逐步去噪生成的样本。它适合一个观测对应多种合理动作、动作具有时间结构和轨迹平滑性的机器人操作任务。

本文负责扩散策略的直觉、条件动作生成、训练/推理流程和工程 Debug；行为克隆基础见 [[Behavior_Cloning]]，扩散模型数学细节属于后续专题，控制执行见 [[Control]]。

## 0. 学习目标

学完以后能够：

- 解释 Diffusion Policy 为什么不是简单回归一个动作；
- 理解 action horizon、noise、denoising 和 condition；
- 读懂训练与采样的基本代码结构；
- 区分 action chunk 与单步动作；
- 调试归一化、时间长度、噪声调度和控制接口；
- 将视觉、状态和语言条件连接到动作生成。

## 1. 为什么需要 Diffusion Policy？

同一观测可能有多种合理轨迹。例如绕开障碍物可以从左边或右边走，抓取动作也可能有多个接近方向。用 MSE 回归平均动作，可能得到谁都不对的中间动作。

```text
observation condition
↓
从噪声动作序列开始
↓
多步 denoising
↓
生成 action chunk
↓
执行前几步
↓
重新观测与生成
```

## 2. 核心概念

### 2.1 Action Chunk

模型一次预测 H 个未来动作，而不是只预测 a_t。这样可表达轨迹相关性和平滑性，但也会增加错误持续时间和推理成本。

### 2.2 Forward Noise

训练时向真实动作序列逐步加入噪声，得到 noisy action。模型学习在给定 observation 和噪声等级下预测噪声、干净动作或速度场。

### 2.3 Denoising

推理时从随机噪声动作开始，反复应用模型，逐渐得到符合条件的动作序列。

### 2.4 Condition

condition 可能包括：

- RGB/depth 视觉 feature；
- 机器人状态；
- 语言指令；
- 目标 pose；
- 历史动作；
- 时间/扩散步 embedding。

## 3. 训练流程

```text
expert action chunk a
↓
随机 diffusion step k
↓
加入噪声 → a_noisy
↓
policy(condition, a_noisy, k)
↓
预测 noise/velocity/clean action
↓
diffusion loss
```

训练目标不是简单把 condition 映射到一个确定动作，而是学习条件动作分布。

## 4. 推理流程

```text
condition
↓
随机初始化 action sequence
↓
k = K...1
↓
denoising network
↓
更新 action sequence
↓
得到 H 步动作
↓
执行前 h 步
↓
receding horizon replan
```

执行 h 步后重新生成，可以利用新观测纠正误差。

## 5. 从零读代码

```python
noise = torch.randn_like(action)
t = torch.randint(0, num_steps, (action.size(0),))
noisy_action = scheduler.add_noise(action, noise, t)

pred_noise = model(
    obs_condition=condition,
    noisy_action=noisy_action,
    diffusion_step=t,
)
loss = mse_loss(pred_noise, noise)
```

要确认：

- action shape 是 [B,H,A] 还是其他布局；
- scheduler 的时间步；
- target 是 noise、velocity 还是 clean action；
- action 和 condition 是否同一时间；
- 推理时 scheduler step 是否正确。

## 6. 归一化与控制

扩散模型通常在归一化动作空间训练。部署时需要：

```text
physical action
↓ normalize
training action space
↓ diffusion policy
normalized action chunk
↓ unnormalize
physical action
↓ limit/controller
robot
```

漏掉 unnormalize 或 action chunk 执行频率错误，会导致真实动作幅度和速度异常。

## 7. 在具身智能中的使用

常见结构：

```text
RGB/depth
↓ CNN/ViT encoder
robot state + language
↓ condition fusion
diffusion action decoder
↓ action chunk
IK/controller
↓ robot
```

它可用于 manipulation、导航、视觉伺服和多模态策略。

## 8. 常见错误与 Debug

### 8.1 action horizon 错

训练 H 步，推理或 controller 按不同长度解释，导致 shape 或时间语义错。

### 8.2 噪声目标错

预测 noise、velocity 和 clean action 不是同一个目标，必须与 scheduler 配套。

### 8.3 动作归一化错

检查 dataset statistics、训练 transform、反归一化和物理 action limits。

### 8.4 采样太慢

扩散步数、视觉 encoder、动作 horizon 和硬件都会影响实时性。具身控制需评估延迟。

### 8.5 生成动作不平滑

检查 action loss、时间条件、chunk 执行、控制频率和轨迹数据；可加入平滑或速度约束。

## 9. 与其他笔记的关系

前置：[[Behavior_Cloning]]、[[视觉表征]]、[[Transformer]]、[[Control]]、[[PyTorch]]。

相关：[[VLA]]、[[Representation_for_Control]]、[[Planning_and_Control]]、[[Robot_Learning]]。

本篇不推导完整 DDPM/DDIM；重点是读懂策略数据流和部署边界。

## 10. 知识地图

```text
trajectory data
↓
normalize
↓
noise schedule
↓
conditional denoising
↓
action chunk
↓
unnormalize
↓
controller
↓
replan
```

## 11. 快速查询表

| 检查项 | 重点 |
|---|---|
| action shape | [B,H,A] |
| condition | 图像、状态、语言、目标 |
| target | noise/velocity/clean |
| scheduler | 训练与推理匹配 |
| horizon | 预测、执行、重规划 |
| normalize | 训练/部署一致 |
| latency | denoise + encoder + control |

## 12. 练习题（题目下附参考答案）

1. Diffusion Policy 解决什么问题？

**参考答案：** 建模条件下的多模态动作序列，而不是只回归一个平均动作。

2. action chunk 是什么？

**参考答案：** 连续多个未来动作组成的动作序列。

3. 训练时为什么加噪声？

**参考答案：** 让模型学习从不同噪声程度逐步恢复合理动作。

4. 推理从什么开始？

**参考答案：** 通常从随机噪声动作序列开始，逐步去噪。

5. condition 可以包括什么？

**参考答案：** 视觉、机器人状态、语言、目标和历史动作。

6. diffusion step 是什么？

**参考答案：** 表示当前噪声/去噪阶段的条件变量。

7. 为什么 MSE 单步回归可能不好？

**参考答案：** 多种合理动作平均后可能得到不合理的中间动作。

8. 训练目标必须和什么匹配？

**参考答案：** scheduler 和模型设定，预测 noise、velocity、clean action 不同。

9. 为什么要做动作归一化？

**参考答案：** 让训练数值尺度稳定，减少不同动作维度量纲差异。

10. 漏掉 unnormalize 会怎样？

**参考答案：** 归一化数值被直接当物理动作执行，幅度和单位错误。

11. receding horizon 是什么？

**参考答案：** 预测一段、执行前几步、重新观测并再次规划。

12. diffusion 推理慢怎么查？

**参考答案：** 检查采样步数、视觉编码、horizon、硬件和控制频率。

13. 生成动作抖动可能为什么？

**参考答案：** 数据轨迹噪声、时间条件、chunk 执行、控制频率或损失不足。

14. 训练 H=16、执行 H=1 可以吗？

**参考答案：** 可以作为滚动执行设计，但要确认 scheduler、重规划和时间对齐。

15. 视觉 policy 中 condition 先经过什么？

**参考答案：** CNN/ViT encoder 得到视觉 feature，再与状态/语言融合。

16. 为什么真实机器人要评估延迟？

**参考答案：** 扩散采样和感知延迟会让动作基于过时观测，影响闭环稳定。

17. Diffusion Policy 与 BC 的关系是什么？

**参考答案：** 都可从示范训练，Diffusion Policy 以生成分布方式表示多种动作轨迹。

18. 如何验证动作 chunk 的时间语义？

**参考答案：** 打印每个预测位置对应的时间戳、执行频率和 controller step。

19. 为什么要加安全限幅？

**参考答案：** 生成模型可能产生越界或异常动作，不能直接绕过硬件保护。

20. 读 Diffusion Policy 代码最先看什么？

**参考答案：** action shape/horizon、noise target、condition、scheduler、归一化和执行循环。

## 13. 掌握度检查（附答案）

1. Diffusion Policy 与确定性回归的差异是什么？

**答案：** 它学习条件动作分布，可产生多个合理轨迹，而非只输出平均动作。

2. action horizon 表示什么？

**答案：** 一次预测的未来动作步数。

3. 训练时加噪、推理时去噪的关系是什么？

**答案：** 训练模拟不同噪声状态，推理从噪声反向恢复动作。

4. 为什么 condition 重要？

**答案：** 它告诉模型当前视觉、状态、目标和语言上下文。

5. scheduler 负责什么？

**答案：** 定义噪声/去噪时间步和更新规则。

6. 为什么需要 normalization？

**答案：** 统一动作尺度并改善训练数值稳定性。

7. 为什么需要 unnormalize？

**答案：** 把模型空间动作还原为真实物理单位。

8. receding horizon 有什么好处？

**答案：** 通过新观测持续纠正生成误差。

9. Diffusion Policy 的部署瓶颈是什么？

**答案：** 采样步数、感知编码、动作 horizon、延迟和控制频率。

10. 为什么生成轨迹要检查平滑性？

**答案：** 机器人动作需要速度/加速度连续，抖动会影响安全和接触。

11. Diffusion Policy 如何用于视觉操作？

**答案：** 视觉 encoder 提供 condition，扩散 decoder 生成动作序列。

12. 为什么目标类型必须确认？

**答案：** 预测 noise、velocity、clean action 对应不同 loss 和 scheduler。

13. 何时查 [[Behavior_Cloning]]？

**答案：** 需要理解示范数据、动作标签、分布偏移和闭环 BC 时。

14. 何时查 [[Planning_and_Control]]？

**答案：** 需要理解 action chunk 如何被执行和控制器如何跟踪时。

15. 读 Diffusion Policy research code 最先确认哪五项？

**答案：** condition、action horizon、noise target、scheduler、physical action mapping。
