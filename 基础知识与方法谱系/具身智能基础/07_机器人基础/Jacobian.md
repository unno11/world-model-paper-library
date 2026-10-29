# Jacobian：把关节变化连接到末端运动

Jacobian（雅可比矩阵）描述机器人关节变量发生小变化时，末端位置和姿态如何变化。它是速度控制、数值 IK、奇异性分析和力控制的桥梁。

微积分中的 Jacobian 见 [[微积分]]，正向运动学见 [[Forward_Kinematics]]，逆向运动学见 [[Inverse_Kinematics]]。

## 0. 学习目标

学完以后能够：

- 解释 Jacobian 的输入输出和 shape；
- 区分位置 Jacobian 与完整 geometric Jacobian；
- 理解 q_dot 到末端 twist 的映射；
- 解释奇异位形和伪逆；
- 读懂 Jacobian-based IK 和速度控制代码；
- 排查 frame、单位、列顺序和数值尺度错误。

## 1. 为什么需要 Jacobian？

FK 给出位置，但控制还要知道“关节速度会让末端怎么动”：

```text
关节位置 q
↓ 微分
末端 pose x
↓
关节速度 q_dot
↓ Jacobian
末端速度 x_dot
```

局部线性关系：

$$
\dot{x}=J(q)\dot{q}
$$

## 2. 核心概念

### 2.1 Jacobian 的 shape

若末端任务空间为 6 维 twist（3 维线速度 + 3 维角速度），关节数为 n：

$$
J\in R^{6\times n}
$$

若只控制位置，可能使用 3×n 的 position Jacobian。

### 2.2 Twist

twist 是末端的速度表示：

$$
\dot{x}=[v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]^T
$$

线速度和角速度必须明确 frame 和单位。

### 2.3 Jacobian 的列

每一列描述一个关节速度对末端 twist 的影响。旋转关节和移动关节的几何形式不同。

### 2.4 奇异性

当 Jacobian 秩下降时，某些末端方向无法通过有限/合理的关节速度产生，或伪逆需要很大 q_dot。奇异附近控制会放大噪声和速度。

## 3. Jacobian 的整体工作原理

```text
当前 q
↓
FK 得各关节轴和末端位置
↓
构造 J(q)
↓
速度控制：q_dot → x_dot
或 IK：x_error → q_dot
↓
积分得到 q_next
```

## 4. 速度控制与伪逆

当希望末端速度为 x_dot_des：

$$
\dot{q}=J^+\dot{x}_{des}
$$

如果是位置控制，可用：

$$
\dot{q}=J^+K(x_{des}-x)
$$

K 是增益。实际系统还需速度限幅、关节限位、阻尼和 null-space 目标。

## 5. Damped Least Squares

阻尼伪逆：

$$
J^T(JJ^T+\lambda^2I)^{-1}
$$

lambda 越大，通常越稳定但跟踪会更保守。阻尼可根据最小奇异值自适应调整。

## 6. Null Space

当关节多于任务维度时，存在不改变末端一阶运动的关节变化。可利用 null space：

- 远离关节限位；
- 避免碰撞；
- 保持舒适姿态；
- 优化能量或可操作性。

不要让次任务破坏主任务约束。

## 7. 从零读 Jacobian 代码

```python
J = robot.jacobian(q)       # [6, n]
x_error = target_twist - current_twist
dq = torch.linalg.pinv(J) @ x_error
dq = torch.clamp(dq, -dq_max, dq_max)
q_next = q + dt * dq
```

检查：

- J 的行顺序是线速度/角速度还是相反；
- J 的 frame；
- x_error 单位；
- q 是 rad 还是 m；
- dt 和速度限幅是否一致；
- pinv 是否在奇异附近稳定。

## 8. 在 AI 与具身智能中的使用

Jacobian 可用于：

- differentiable robotics；
- IK solver；
- 末端速度控制；
- policy action 从 Cartesian 转 joint；
- model-based planning；
- 关节动作正则化；
- 训练神经策略的几何约束。

学习型 policy 也可以预测 q_dot，但 Jacobian 提供一种可解释的几何基线和安全过滤器。

## 9. 常见错误与 Debug

### 9.1 Jacobian frame 错

body Jacobian、spatial Jacobian 或不同库定义可能不同。速度和误差必须在匹配 frame。

### 9.2 行列含义颠倒

J 是 [6,n] 还是 [n,6]？打印 shape，并用单关节数值差分验证一列。

### 9.3 角速度单位混合

线速度 m/s、角速度 rad/s 不能不加权直接比较。控制 loss 或阻尼需要尺度处理。

### 9.4 奇异附近 q_dot 爆炸

使用阻尼、速度限制、路径绕开奇异位形或增加 null-space 目标。

### 9.5 dt 使用错误

q_next=q+dt*q_dot；dt 错会让动作速度异常。

## 10. 与其他笔记的关系

前置：[[微积分]]、[[线性代数]]、[[坐标系与变换]]、[[旋转表示]]、[[Forward_Kinematics]]。

相关：[[Inverse_Kinematics]]、[[Dynamics]]、[[Control]]、[[Manipulation]]。

本篇不展开完整动力学控制器，重点是局部几何映射。

## 11. 知识地图

```text
q
↓ FK
J(q)
├─ q_dot → x_dot
├─ x_error → q_dot
└─ singularity / null space
↓
速度、IK、控制和安全
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| shape | [6,n] 或 [n,6] |
| rows | v/omega 顺序 |
| frame | body/spatial/base |
| units | m/s、rad/s |
| singularity | rank、condition number |
| update | q+dt*q_dot |
| safety | damping、clamp、limits |

## 13. 练习题（题目下附参考答案）

1. Jacobian 描述什么？

**参考答案：** 描述关节变量变化与末端局部运动之间的关系。

2. 完整 6D Jacobian 的 shape 是什么？

**参考答案：** 末端 twist 6 维、关节 n 个时为 [6,n]。

3. twist 包含什么？

**参考答案：** 3 维线速度和 3 维角速度。

4. Jacobian 的每一列代表什么？

**参考答案：** 一个关节速度对末端 twist 的影响。

5. 速度映射公式是什么？

**参考答案：** x_dot=J(q) q_dot。

6. 伪逆用于什么？

**参考答案：** 从期望末端速度或误差求近似关节速度。

7. 奇异位形有什么风险？

**参考答案：** 某些方向不可控或需要巨大关节速度，噪声也会被放大。

8. 阻尼伪逆解决什么？

**参考答案：** 改善奇异附近的数值稳定性。

9. 为什么要区分 Jacobian frame？

**参考答案：** 速度分量在不同 frame 中数值和方向不同，必须与误差匹配。

10. q_next 的基本积分公式是什么？

**参考答案：** q_next=q+dt*q_dot。

11. 为什么要限制 q_dot？

**参考答案：** 防止奇异、噪声或大误差导致超速和不安全动作。

12. null space 可以优化什么？

**参考答案：** 关节限位、碰撞、姿态舒适度和能量等次目标。

13. 如何用数值差分检查 Jacobian 一列？

**参考答案：** 对单个关节加小扰动，用 FK 计算末端差分并与对应列比较。

14. 行列顺序错会怎样？

**参考答案：** 控制方向和维度对应错误，可能运行但运动完全不对。

15. 线速度和角速度为什么要加权？

**参考答案：** 单位和任务重要性不同，不能直接无尺度比较。

16. Jacobian 如何帮助 IK？

**参考答案：** 把 pose error 近似转换为关节增量。

17. policy 直接输出 q_dot 时 Jacobian 有什么用？

**参考答案：** 可作几何基线、可操作性约束或安全过滤器。

18. 奇异附近控制出现抖动怎么查？

**参考答案：** 检查奇异值/条件数、阻尼、速度限幅、dt 和 frame。

19. 为什么 Jacobian 依赖 q？

**参考答案：** 机器人几何和关节轴在不同配置下对末端的影响不同。

20. 读 Jacobian 代码最先看什么？

**参考答案：** shape、行列语义、frame、单位、伪逆/阻尼和积分步长。

## 13. 掌握度检查（附答案）

1. Jacobian 的输入输出是什么？

**答案：** 输入关节速度，输出末端线速度/角速度的局部线性关系。

2. J 的列数代表什么？

**答案：** 关节变量数量。

3. J 的行数 6 代表什么？

**答案：** 3 个线速度分量和 3 个角速度分量。

4. 什么是奇异性？

**答案：** Jacobian 秩下降或接近秩下降，某些任务方向难以产生。

5. 为什么伪逆会爆炸？

**答案：** 小奇异值被求逆后放大误差和噪声。

6. 阻尼的代价是什么？

**答案：** 稳定性提高，但跟踪可能更保守、误差更大。

7. null space 用来做什么？

**答案：** 在不影响主任务的自由度上优化次目标。

8. 为什么 dt 重要？

**答案：** 速度积分成位置需要正确时间步，dt 错会改变实际动作幅度。

9. frame 错误如何发现？

**答案：** 用已知方向/单关节运动对比预测和实际末端速度。

10. Jacobian 与 FK 的关系是什么？

**答案：** Jacobian 是 FK 对关节变量的局部导数。

11. Jacobian 与 IK 的关系是什么？

**答案：** 用 J 或其伪逆把末端误差映射为关节更新。

12. Jacobian 与控制的关系是什么？

**答案：** 将空间速度/误差转换为关节速度命令。

13. 何时查 [[微积分]]？

**答案：** 需要理解 Jacobian 作为多变量导数时。

14. 何时查 [[Inverse_Kinematics]]？

**答案：** 需要用 Jacobian 迭代求目标 pose 时。

15. 读 Jacobian research code 最先确认哪五项？

**答案：** 输入 q、shape/行列、frame、单位、奇异和限速处理。
