![Pasted image 20260726142311.png](../../%E5%9B%BE%E7%89%87%E5%BC%95%E7%94%A8/d39aced5c3-Pasted%20image%2020260726142311.png)
JEPA-WM 的完整数据流可以拆成两个闭环：训练阶段学习“动作会让世界的表征如何变化”，规划阶段利用这个模型反向搜索“应该执行什么动作”。
## 一、原始数据

Wall、PointMaze 等数据集提供的是离线轨迹：

```
trajectory =
{
    图像观测:       o0, o1, o2, ...
    动作:           a0, a1, a2, ...
    本体状态:       p0, p1, p2, ...    # 可选
}
```

其中：

- `o_t`：第 t 时刻的 RGB 图像；
- `a_t`：机器人或二维 agent 执行的动作；
- `p_t`：位置、速度、关节状态等 proprioception；
- 不需要 reward；
- 不需要人工标注“这个动作好不好”。

模型只需要从已有轨迹中学习：

```
当前状态 + 动作 → 未来状态
```

## 二、视觉数据流

图像经过预处理后进入冻结的 DINOv2：

```
RGB 图像
  ↓ resize / normalize / augmentation
DINOv2 ViT-S/14
  ↓
空间 patch tokens
z_t^vis
```

DINOv2 不直接输出一个类别，也不是在这里做分类。它把图像变成带有空间位置关系的 patch features，例如：

```
墙的位置
agent 的位置
目标的位置
物体的局部结构
```

这些视觉特征就是世界模型的“视觉状态”。

关键点是：

> 当前训练通常不更新 DINOv2，只训练其上方的动力学 predictor。

因此，DINOv2 相当于“感知系统”，JEPA-WM predictor 相当于“动力学模型”。

## 三、本体状态数据流

如果配置启用 proprioception：

```
位置、速度、关节状态
        ↓
proprio encoder
        ↓
z_t^prop
```

它补充视觉图像中不容易精确判断的信息，例如：

- Wall/PointMaze 中 agent 的精确位置；
- Metaworld 中机械臂末端位置；
- 速度和运动方向；
- 距离目标还有多远。

最终状态可以理解为：

```
z_t = {
    visual latent,
    proprio latent
}
```

但在具体实现中，它们可能以 token、feature conditioning 或 AdaLN conditioning 的方式进入 predictor，并不一定直接拼成一个向量。

## 四、动作数据流

动作首先经过 action encoder：

```
原始动作 a_t
    ↓
action encoder
    ↓
action embedding e_t^a
```

动作 embedding 告诉 predictor：

> 在当前状态下，如果执行这个动作，未来状态应该怎样变化？

当前 JEPA-WM 使用 AdaLN，把动作条件注入 predictor 的多个 Transformer block，而不是只在输入端提供一次。

这能减少网络加深后动作信息逐渐消失的问题。

## 五、Predictor 的输入

模型不会只看单帧，而是读取一个时间上下文：

```
z_{t-W+1}, ..., z_t
a_{t-W+1}, ..., a_t
```

当前论文 simulation 配置中，最大训练上下文通常是：

```
W = 3
```

它的意义是：

- 一帧只能看到位置；
- 两帧可以推断速度；
- 三帧可以进一步推断加速度或运动趋势。

Predictor 的核心计算可以简化成：

```
ẑ_{t+1} = P(
    历史状态 latent,
    历史与当前动作
)
```

这里输出的是未来 latent，不是未来 RGB 图像。

## 六、Teacher forcing 数据流

最基础的一步预测是：

```
真实 z_t + 动作 a_t
        ↓
Predictor
        ↓
预测 ẑ_{t+1}
        ↓ 比较
真实 z_{t+1}
```

损失大致是：

```
L1 = MSE(ẑ_{t+1}, z_{t+1})
```

其中真实目标 `z_{t+1}` 也是由冻结的 DINOv2 对下一帧编码得到的。

因此模型学习的是：

> 让预测的未来视觉表征接近真实未来视觉表征。

## 七、2-step rollout 数据流

如果永远给 predictor 真实历史 latent，规划时会出现训练和推理不一致：

```
训练：每一步输入都是真实 latent
规划：下一步输入是模型自己的预测 latent
```

所以 JEPA-WM 加入 2-step rollout：

```
真实 z_t
  ↓ P(z_t, a_t)
预测 ẑ_{t+1}
  ↓ P(ẑ_{t+1}, a_{t+1})
预测 ẑ_{t+2}
```

然后分别和真实目标比较：

```
L = L1 + L2
```

这就是配置名里的：

```
repro_2roll
```

它让模型学会处理自己的预测误差，降低多步展开时的误差积累。

## 八、反向传播

损失计算完成后：

```
latent prediction loss
        ↓ backward
更新 Predictor
更新 action encoder
更新 proprio encoder
不更新 DINOv2 visual encoder
```

在你们的 4-GPU 环境中：

```
4 个 GPU rank
  ↓ 各自计算 minibatch gradient
gradient accumulation = 4
  ↓
gloo 同步梯度
  ↓
optimizer.step()
```

训练完成后 checkpoint 主要保存：

- predictor；
- action encoder；
- proprio encoder；
- optimizer/epoch 等训练状态；
- 通常不重复保存完整 DINOv2 权重，或只保存其相关状态引用。

## 九、规划阶段的输入

规划时有两个核心观测：

```
当前图像 o_t
目标图像 o_g
```

分别编码：

```
z_t = E(o_t)
z_g = E(o_g)
```

目标并不是通过 reward 表达，而是通过目标图像的 latent 表达。

所以 JEPA-WM 做的是 goal-conditioned planning：

> 找到一串动作，使预测的最终 latent 尽量靠近目标图像 latent。

## 十、CEM 动作搜索

CEM 首先随机生成很多候选动作序列：

```
候选 1: a_t, a_{t+1}, ..., a_{t+H-1}
候选 2: a_t, a_{t+1}, ..., a_{t+H-1}
...
候选 N
```

每条候选动作都送入 predictor rollout：

```
z_t
 + candidate actions
        ↓
ẑ_{t+1}, ẑ_{t+2}, ..., ẑ_{t+H}
```

然后计算最终预测状态到目标状态的距离：

```
cost =
    L2(ẑ_{t+H}^vis, z_g^vis)
  + α × L2(ẑ_{t+H}^prop, z_g^prop)
```

当前 simulation 配置通常使用：

```
planner = CEM
distance = L2
α = 0.1       # 启用 proprioception 时
```

## 十一、CEM 如何优化动作

CEM 不直接使用神经网络梯度修改动作，而是：

```
采样大量动作序列
        ↓
用 JEPA-WM 预测结果
        ↓
计算每条轨迹的 cost
        ↓
选择 cost 最低的 elite trajectories
        ↓
更新动作分布的均值和方差
        ↓
重新采样
```

反复迭代后得到最优动作序列。

这也是为什么模型的 prediction loss 很低，不代表 planning success 一定高：

- Predictor 负责预测；
- CEM 负责搜索；
- planning cost 决定什么叫“接近目标”；
- 三者必须共同工作。

## 十二、MPC 执行闭环

JEPA-WM 通常不会一次执行完整计划，而是使用 Model Predictive Control：

```
当前观测
  ↓
规划 H 步动作
  ↓
只执行前 m 步
  ↓
环境返回新观测
  ↓
重新编码、重新规划
```

这样即使世界模型预测有误，也可以利用真实新观测不断纠正。

完整闭环是：

```
感知 → latent
  → 候选动作采样
  → 世界模型 rollout
  → latent cost
  → CEM 优化
  → 执行动作
  → 新感知
  → 再规划
```

## 十三、训练和评估输出的区别

训练日志中的：

```
final loss = 0.008
```

只说明：

```
预测 latent 与真实 latent 比较接近
```

它不等于：

```
Wall planning success = 78.8%
```

真正验证 baseline 必须继续执行：

```
训练 checkpoint
    ↓
planning eval
    ↓
96 个独立 episodes
    ↓
success_count / success_rate
```