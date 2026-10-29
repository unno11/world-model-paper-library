# Control：让机器人稳定地执行目标运动

机器人控制（Robot Control）负责把目标位置、速度、力或动作转换为执行器命令，并通过反馈不断纠正误差。它连接感知、运动学、动力学与真实硬件。

本文重点负责开环/闭环、位置/速度/力控制、PID、阻抗控制和控制循环；动力学见 [[Dynamics]]，Jacobian 见 [[Jacobian]]，策略决策见 [[MDP]]、[[Actor_Critic]]。

## 0. 学习目标

学完以后能够：

- 区分开环控制与闭环控制；
- 理解误差、反馈、PID 和控制频率；
- 读懂 joint-space 与 task-space 控制；
- 解释 impedance/admittance 的基本直觉；
- 连接 IK、动力学和 policy action；
- 排查单位、延迟、饱和、振荡和安全问题。

## 1. 为什么需要控制？

规划器或策略只给出目标，真实机器人不会瞬间到达。执行器有惯性、摩擦、延迟和扰动，因此需要反馈：

```text
目标 x_des
↓
与当前 x 比较
↓
误差 e
↓
控制器
↓
电机/执行器命令
↓
机器人运动
↓
传感器反馈
└───────────────
```

## 2. 核心概念

### 2.1 Open-Loop 与 Closed-Loop

- open-loop：只按预先命令执行，不看反馈；
- closed-loop：根据当前误差持续修正。

真实机器人通常必须闭环，尤其在视觉和接触任务中。

### 2.2 Position、Velocity、Torque Control

- position control：目标关节位置；
- velocity control：目标速度；
- torque control：目标力矩；
- task-space control：目标末端位置/姿态或力。

接口名字不一定代表底层真实控制模式，要看硬件和 SDK。

### 2.3 PID

误差 e(t)：

```text
P：当前误差
I：误差累积
D：误差变化速度
u = Kp e + Ki ∫e dt + Kd de/dt
```

- P 提供纠正；
- I 消除持续偏差；
- D 抑制快速变化，但对噪声敏感。

机器人常用离散实现，积分要防 windup。

### 2.4 控制频率与延迟

控制器每秒运行多少次、传感器延迟多久、命令何时真正执行，都会影响稳定性。频率和延迟必须与模型和数据采集对齐。

### 2.5 饱和与安全

执行器有位置、速度、力矩和电流限制。控制输出超限后会被 clip，导致实际闭环与理论模型不同。安全层应限制动作、监测异常并支持急停。

## 3. 控制循环

```text
读取传感器
↓
估计当前状态
↓
计算目标误差
↓
控制器输出命令
↓
限幅/安全检查
↓
发送执行器
↓
等待 dt
↓
重复
```

每一步都要记录 timestamp、state、target、error 和 command，便于 Debug。

## 4. 从零读 PID 代码

```python
error = target - position
integral = integral + error * dt
derivative = (error - prev_error) / dt

command = kp * error + ki * integral + kd * derivative
command = torch.clamp(command, -limit, limit)

prev_error = error
```

检查：

- target 和 position 单位；
- dt 是否真实；
- derivative 是否受噪声影响；
- integral 是否需要限幅；
- command 是否与硬件接口匹配。

## 5. Task-Space 控制

末端位姿误差可以通过 Jacobian 转成关节速度/力矩：

```text
末端目标 pose
↓
pose error
↓
task-space controller
↓ Jacobian / dynamics
joint command
↓
robot
```

task-space 控制更贴近抓取和插入任务，但需要正确 frame、姿态误差和动力学/雅可比。

## 6. Impedance 与 Admittance

- impedance：给定运动误差产生期望力，像弹簧—阻尼系统；
- admittance：给定外力产生期望运动。

接触任务中，刚性位置控制可能造成冲击；阻抗/导纳让机器人更柔顺，但参数错误也会导致不稳定或下陷。

## 7. Policy 与低层控制

高层 policy 可能输出：

- 关节位置/速度增量；
- 末端 delta pose；
- 目标 waypoint；
- skill command。

低层控制器负责高频跟踪和安全限制。不要把神经 policy 的输出直接等同于电机电流，除非系统明确如此设计。

## 8. 在具身智能中的使用

```text
视觉/状态 observation
↓
policy 或 planner
↓
高层 action
↓
IK / Jacobian / controller
↓
低层执行器
↓
新 observation
```

这条链路决定 research code 能否变成真实动作。离线 policy loss 低，但控制频率、延迟和动作接口错了，机器人仍会失败。

## 9. 常见错误与 Debug

### 9.1 P 增益太大

动作快速、振荡、噪声放大。逐步降低增益并检查控制频率和延迟。

### 9.2 积分 windup

输出饱和时积分继续累积，解除饱和后会严重过冲。使用积分限幅或 anti-windup。

### 9.3 derivative 对噪声敏感

对测量值直接差分会放大噪声，可滤波并检查 dt。

### 9.4 单位和 frame 错

度/弧度、mm/m、局部/全局坐标错误会导致命令方向和幅度异常。

### 9.5 控制频率不匹配

训练 policy 的 action interval 与部署不同，动作平滑和稳定性会改变。

### 9.6 没有安全层

任何学习策略都应有动作限幅、碰撞/力监测、回退和急停，不应直接绕过保护。

## 10. 与其他笔记的关系

前置：[[坐标系与变换]]、[[旋转表示]]、[[Dynamics]]、[[Jacobian]]、[[PyTorch]]。

相关：[[Inverse_Kinematics]]、[[Manipulation]]、[[MDP]]、[[Actor_Critic]]、[[Imitation_Learning]]。

本篇不展开具体机器人厂商控制接口；重点是控制思想和 research code 到硬件的边界。

## 11. 知识地图

```text
目标
↓
误差
↓
反馈控制
├─ position
├─ velocity
├─ torque
└─ impedance/admittance
↓
限幅与安全
↓
执行器
↓
传感器反馈
```

## 12. 快速查询表

| 检查项 | 重点 |
|---|---|
| target/state | frame、单位 |
| error | 位置/姿态/力 |
| dt | 实际控制周期 |
| gains | 振荡/响应速度 |
| saturation | clip 与 anti-windup |
| delay | 感知/计算/执行 |
| safety | 限制、监控、急停 |

## 13. 练习题（题目下附参考答案）

1. 闭环控制是什么？

**参考答案：** 使用传感器反馈，根据当前误差持续修正控制命令。

2. PID 的 P、I、D 分别代表什么？

**参考答案：** 比例、积分、微分。

3. 为什么真实机器人常需要闭环？

**参考答案：** 存在扰动、模型误差、延迟、摩擦和执行误差，开环不能自动纠正。

4. 控制频率为什么重要？

**参考答案：** 它决定反馈更新速度、离散积分和稳定性。

5. torque control 与 position control 有何区别？

**参考答案：** 一个直接给力矩，一个给位置目标，底层执行与柔顺性不同。

6. P 增益太大可能怎样？

**参考答案：** 响应快但容易振荡、过冲和放大噪声。

7. integral windup 是什么？

**参考答案：** 输出饱和时积分继续累积，解除饱和后产生严重过冲。

8. 为什么 derivative 需要关注噪声？

**参考答案：** 差分会放大高频测量噪声。

9. policy 输出 delta pose 后通常经过什么？

**参考答案：** 结合当前 pose 得目标，再通过 IK/Jacobian/控制器转换为执行命令。

10. task-space 控制的优点是什么？

**参考答案：** 目标直接用末端位置/姿态描述，更贴近抓取和插入任务。

11. impedance control 的直觉是什么？

**参考答案：** 用类似弹簧—阻尼的关系让机器人对误差产生柔顺力。

12. 为什么要限幅 command？

**参考答案：** 保护执行器和机器人，避免异常策略输出造成危险。

13. 控制抖动时排查什么？

**参考答案：** 增益、dt、延迟、噪声、量化、单位、姿态误差和 action 频率。

14. 位置误差一直不为零可能为什么？

**参考答案：** 摩擦、重力、积分不足、目标不可达、偏置或模型误差。

15. 训练和部署 action interval 不同会怎样？

**参考答案：** 动作幅度、速度和稳定性改变，policy 的时间语义不再一致。

16. 为什么高层 policy 不应直接绕过安全控制？

**参考答案：** 学习策略可能输出越界或异常命令，需要低层限幅、监控和急停。

17. 视觉延迟如何影响控制？

**参考答案：** 控制器依据过时状态，可能产生过冲、追赶和不稳定。

18. 力控制适合什么任务？

**参考答案：** 插入、接触、打磨和需要控制接触力的任务。

19. 如何记录控制 Debug 数据？

**参考答案：** 保存时间戳、目标、状态、误差、命令、限幅状态、传感器和安全事件。

20. 读机器人控制代码最先看什么？

**参考答案：** 控制频率、输入输出单位/frame、误差定义、限幅、延迟和安全接口。

## 14. 掌握度检查（附答案）

1. 开环和闭环的区别是什么？

**答案：** 开环不使用反馈，闭环根据反馈误差持续修正。

2. PID 三项分别解决什么直觉问题？

**答案：** P 看当前误差，I 看累计偏差，D 看误差变化趋势。

3. 为什么 dt 是控制变量的一部分？

**答案：** 离散积分和微分都依赖真实时间步。

4. 什么是饱和？

**答案：** 控制命令超过执行器能力后被限制。

5. windup 为什么产生？

**答案：** 饱和时积分项仍累积。

6. task-space 与 joint-space 控制有何差异？

**答案：** 前者在末端空间定义目标，后者在关节变量空间定义目标。

7. Jacobian 如何帮助 task-space 控制？

**答案：** 把末端速度/误差映射为关节速度。

8. impedance 和 admittance 的方向差异是什么？

**答案：** impedance 从运动误差到力，admittance 从外力到运动。

9. 为什么 action scaling 重要？

**答案：** policy 输出数值与物理位置/速度/力矩之间需要明确映射。

10. 控制器如何连接高层策略？

**答案：** 策略产生目标或增量，低层控制器高频、安全地跟踪。

11. 为什么离线 loss 不能替代控制评估？

**答案：** 控制还受时间、延迟、限幅、闭环误差和硬件影响。

12. 如何排查振荡？

**答案：** 检查增益、频率、延迟、噪声、饱和和模型不匹配。

13. 何时查 [[Dynamics]]？

**答案：** 需要理解重力、惯性、摩擦、接触和前馈力矩时。

14. 何时查 [[Manipulation]]？

**答案：** 需要把控制放入抓取、接触和操作任务时。

15. 读控制 research code 最先确认哪五项？

**答案：** loop rate、目标与误差、单位/frame、限幅/安全、执行与反馈延迟。
