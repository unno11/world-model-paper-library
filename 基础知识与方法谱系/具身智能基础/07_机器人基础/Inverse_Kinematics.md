# Inverse Kinematics：从目标位姿求关节配置

逆向运动学（Inverse Kinematics，IK）回答：给定末端目标位置和姿态，机器人各关节应该取什么值？

它是机器人抓取、导航、操作和轨迹执行的核心。正向运动学见 [[Forward_Kinematics]]，局部速度关系见 [[Jacobian]]，控制执行见 [[Control]]。

## 0. 学习目标

学完以后能够：

- 解释 IK 的输入、输出和多解问题；
- 区分解析 IK、数值 IK 和学习型 IK；
- 理解位置/姿态误差、雅可比伪逆和阻尼；
- 判断目标是否可达、是否接近奇异位形；
- 读懂 IK solver 的停止条件和约束；
- 调试 frame、初值、关节限位、单位和碰撞问题。

## 1. 为什么需要 IK？

任务通常给出空间目标，而机器人执行关节命令：

```text
目标位置/姿态 T_target
↓
IK solver
↓
关节配置 q_target
↓
FK 验证
↓
规划 / 控制
↓
机器人执行
```

一个末端 pose 可能对应多个 q，也可能没有解。IK solver 不只是“解一个方程”，还要考虑限位、碰撞、连续性和任务姿态。

## 2. 核心概念

### 2.1 IK 问题

寻找 q，使：

$$
T(q)\approx T_{target}
$$

位置误差和姿态误差通常组合成 cost：

$$
L(q)=w_p||p(q)-p^*||^2+w_Rd_R(R(q),R^*)
$$

### 2.2 解析 IK

通过几何和代数直接推导解。速度快、可解释，但依赖机器人结构，复杂机械臂难以推导，且要处理多个分支。

### 2.3 数值 IK

从初始 q 出发迭代减少 pose error：

```text
q_0
↓
FK(q_0)
↓
pose error
↓
Jacobian
↓
Δq
↓
q ← q+Δq
↓
直到收敛或失败
```

初值影响很大，可能落入不同解或局部最小。

### 2.4 可达性与工作空间

目标在机器人工作空间外时无解；即使位置可达，姿态也可能不可达。工作空间受连杆长度、关节范围和碰撞约束影响。

### 2.5 约束

实际 IK 可能包含：

- joint limits；
- velocity/acceleration limits；
- collision avoidance；
- 姿态偏好；
- 与上一时刻 q 的连续性；
- 多臂或工具约束。

## 3. Jacobian 伪逆更新

局部关系：

$$
\Delta x=J(q)\Delta q
$$

简单更新：

$$
\Delta q=J^+\Delta x
$$

其中 J^+ 是伪逆。接近奇异位形时伪逆可能放大噪声，常用阻尼最小二乘：

$$
\Delta q=J^T(JJ^T+\lambda^2I)^{-1}\Delta x
$$

Jacobian 细节见 [[Jacobian]]。

## 4. 从零读数值 IK 伪代码

```python
q = q_init.clone()

for _ in range(max_iters):
    pose = fk(q)
    error = pose_error(pose, target_pose)

    if torch.linalg.vector_norm(error) < tol:
        break

    J = jacobian(q)
    dq = damped_pseudoinverse(J, damping) @ error
    q = clamp(q + step_size * dq, q_min, q_max)
```

需要检查：

- pose_error 的位置/姿态定义；
- J 的 frame 与 error 一致；
- dq 单位；
- clamp 是否造成停滞；
- 迭代是否真正降低 error；
- 是否检查碰撞。

## 5. 多解与初值

同一目标可能有肘上/肘下、腕部翻转等多个解。选择解时可以考虑：

- 与当前 q 最近；
- 不接近关节限位；
- 不碰撞；
- 动作更平滑；
- 更符合工具姿态和任务约束。

不要只选择数学误差最小的解。

## 6. 在具身智能中的使用

IK 出现在：

- 视觉目标转末端动作；
- imitation learning 的 pose-to-joint 转换；
- policy 输出末端 delta pose；
- 高层规划生成关节目标；
- differentiable IK；
- Model-Based RL 的动作约束；
- manipulation 数据标注。

真实机器人中常用“笛卡尔目标 → IK → 低层 joint controller”，也可能直接学习 joint policy。

## 7. 常见错误与 Debug

### 7.1 pose frame 不一致

目标在 camera frame，IK 却以 base frame 解释。先做坐标变换并明确目标 frame。

### 7.2 位置/姿态误差尺度失衡

米和弧度直接相加会使某一项压制另一项。检查权重和单位。

### 7.3 初值不合理

初值离目标太远或处在奇异附近，可能不收敛。使用当前 q、多个初值或分段轨迹。

### 7.4 只检查位置不检查姿态

抓取、插入和工具操作常需要方向正确。分别打印 position error 和 rotation error。

### 7.5 IK 解穿过碰撞

数学可行不代表物理安全。加入碰撞检查和路径规划。

### 7.6 clamp 造成不收敛

如果 q 长期贴在限位，solver 可能无法继续。记录每个关节是否触碰边界。

## 8. 与其他笔记的关系

前置：[[坐标系与变换]]、[[旋转表示]]、[[Forward_Kinematics]]、[[线性代数]]。

相关：[[Jacobian]]、[[Dynamics]]、[[Control]]、[[Manipulation]]、[[Model_Based_RL]]。

本篇不详细讲具体机械臂解析解；重点是 IK 问题结构、数值方法和工程验证。

## 9. 知识地图

```text
target pose
↓
frame/unit normalization
↓
pose error
↓
FK + Jacobian
↓
numerical update
↓
limits/collision/continuity
↓
FK validation
↓
control
```

## 10. 快速查询表

| 检查项 | 重点 |
|---|---|
| target frame | base/ee/camera 是否一致 |
| error | position + orientation |
| initial q | 当前姿态/多初值 |
| Jacobian | frame、shape、奇异性 |
| limits | clamp、速度、碰撞 |
| convergence | error、迭代、停止原因 |
| validation | FK 回代 target |

## 11. 练习题（题目下附参考答案）

1. IK 的输入输出是什么？

**参考答案：** 输入目标末端 pose，输出满足目标的关节配置 q。

2. IK 一定有唯一解吗？

**参考答案：** 不一定，可能多解、无解或有连续解集合。

3. 解析 IK 与数值 IK 的区别是什么？

**参考答案：** 解析法直接推导解，数值法从初值迭代优化误差。

4. 为什么需要 FK 验证 IK？

**参考答案：** 把求得的 q 再送入 FK，检查实际 pose 是否接近目标。

5. Jacobian 在数值 IK 中做什么？

**参考答案：** 近似关节变化对末端位姿变化的局部映射，用于计算 Δq。

6. 伪逆方法的风险是什么？

**参考答案：** 接近奇异位形时可能放大噪声，产生巨大关节更新。

7. 阻尼最小二乘解决什么？

**参考答案：** 缓解奇异附近伪逆不稳定，牺牲少量精确性换取稳定性。

8. 为什么初值重要？

**参考答案：** 数值 solver 可能收敛到不同解或局部最小，初值影响结果。

9. IK 目标为什么要包含关节限位？

**参考答案：** 数学上可行的 q 可能超出机械范围或不安全。

10. 位置误差和姿态误差为什么需要权重？

**参考答案：** 单位和任务重要性不同，直接相加会导致尺度失衡。

11. IK 解误差很小但机器人碰撞，说明什么？

**参考答案：** IK 只满足几何目标，没有处理碰撞和路径安全约束。

12. IK 不收敛时先检查什么？

**参考答案：** frame、单位、目标可达性、初值、Jacobian、误差和限位。

13. 为什么要选择离当前 q 最近的解？

**参考答案：** 减少跳变和不必要的大动作，提高轨迹连续性。

14. 相机目标如何输入 IK？

**参考答案：** 先把目标从 camera frame 变换到 IK 使用的 base/control frame。

15. policy 输出 delta pose 之后发生什么？

**参考答案：** 结合当前末端 pose 得目标 pose，再通过 IK 转为关节目标或速度。

16. clamp 后 q 不再变化怎么查？

**参考答案：** 检查是否卡在 joint limit、更新方向、步长和目标是否不可达。

17. 为什么只优化位置可能抓取失败？

**参考答案：** 夹爪方向、接近方向和姿态同样决定抓取是否成功。

18. 多初值 IK 有什么用途？

**参考答案：** 尝试不同解分支，提高找到可行且无碰撞解的概率。

19. World Model 规划为什么可能需要 IK？

**参考答案：** 高层规划输出空间动作/目标，IK 将其转换为可执行关节配置。

20. 读 IK solver 代码最先看什么？

**参考答案：** target frame、pose error、FK/Jacobian、初值、限位/碰撞、停止条件和 FK 回验。

## 12. 掌握度检查（附答案）

1. IK 要解决什么方程？

**答案：** 找到 q，使 FK(q) 的末端 pose 接近目标 pose。

2. 为什么 IK 可能多解？

**答案：** 机械臂不同关节配置可以产生相同末端 pose。

3. 为什么可能无解？

**答案：** 目标超出工作空间、姿态不可达或约束冲突。

4. 数值 IK 依赖什么？

**答案：** 初值、误差定义、FK、Jacobian、步长、限位和停止条件。

5. Jacobian 伪逆有什么问题？

**答案：** 奇异附近可能放大噪声和关节速度。

6. 阻尼的作用是什么？

**答案：** 改善奇异附近的数值稳定性。

7. 为什么要检查 position 和 orientation 两种误差？

**答案：** 任务往往同时要求位置和工具方向。

8. 为什么目标 frame 必须一致？

**答案：** IK 的误差和 Jacobian 必须在同一空间参考系解释。

9. 为什么需要碰撞检查？

**答案：** 几何可达不等于运动路径安全。

10. 为什么选择离当前 q 最近的解？

**答案：** 让运动连续并降低跳变。

11. IK 如何连接到视觉操作？

**答案：** 视觉目标经坐标变换进入 IK，再转为关节动作。

12. IK 如何连接到控制？

**答案：** IK 产生位置/速度/关节目标，控制器跟踪这些目标。

13. 何时查 [[Jacobian]]？

**答案：** 需要理解速度映射、奇异性和数值 IK 更新时。

14. 何时查 [[Control]]？

**答案：** 需要把 IK 目标变成稳定的执行命令时。

15. 读 IK research code 最先确认哪五项？

**答案：** target frame、误差表示、Jacobian、约束/停止、FK 回代验证。
