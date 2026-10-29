# Forward Kinematics：从关节角计算末端位姿

正向运动学（Forward Kinematics，FK）回答：已知机器人每个关节的位置，末端执行器在某个参考坐标系中的位置和姿态是什么？

它是机器人仿真、可视化、碰撞检测、控制和数据标注的基础。坐标与变换见 [[坐标系与变换]]，逆向求关节见 [[Inverse_Kinematics]]。

## 0. 学习目标

学完以后能够：

- 解释关节、连杆、末端和 kinematic chain；
- 读懂由多个关节变换相乘得到的末端 pose；
- 区分 revolute 与 prismatic joint；
- 理解 DH 参数的作用；
- 通过 FK 验证关节状态和 frame 定义；
- 排查轴方向、零位、单位和变换顺序错误。

## 1. 为什么需要 FK？

控制器知道的是关节角，但任务通常描述末端目标：

```text
joint positions q
↓
每个 joint/link 的局部变换
↓
沿机器人链相乘
↓
末端 pose T_base←ee
↓
可视化、碰撞检查、控制
```

没有 FK，无法知道当前夹爪是否到达目标，也无法把关节状态转换成空间轨迹。

## 2. 核心概念

### 2.1 Joint 与 Link

- joint：允许相邻连杆相对运动的关节；
- link：连接关节的刚体部分；
- kinematic chain：从 base 到 end-effector 的有序连接；
- q：所有关节变量组成的向量。

### 2.2 Revolute 与 Prismatic

- revolute：绕轴旋转，变量通常是角度；
- prismatic：沿轴平移，变量通常是长度。

### 2.3 末端 Pose

FK 输出通常是末端相对于 base/world 的位置和旋转：

$$
T_{base\leftarrow ee}(q)
$$

它是 q 的函数。

### 2.4 变换链

```text
T_base←link1
× T_link1←link2
× ...
× T_linkN←ee
= T_base←ee
```

每个局部变换包含固定结构参数和当前关节变量。

## 3. DH 参数直觉

Denavit-Hartenberg（DH）参数用一组规则描述相邻关节 frame 的关系，常包括：

- a：连杆长度；
- alpha：连杆扭转；
- d：沿轴偏移；
- theta：绕轴角度。

不同库可能使用标准 DH 或 modified DH，必须确认约定。

## 4. 一个二维平面双连杆例子

两根长度 l1、l2 的旋转连杆：

$$
x=l_1\cos q_1+l_2\cos(q_1+q_2)
$$

$$
y=l_1\sin q_1+l_2\sin(q_1+q_2)
$$

第二根连杆的绝对方向是 q1+q2，而不是只有 q2。

## 5. 从零读 FK 代码

```python
def planar_fk(q, l1, l2):
    q1, q2 = q
    x = l1 * torch.cos(q1) + l2 * torch.cos(q1 + q2)
    y = l1 * torch.sin(q1) + l2 * torch.sin(q1 + q2)
    return torch.stack([x, y])
```

代码里 q 是弧度，l1/l2 与输出使用相同长度单位。若 q 是 batch，shape 和广播要另外确认。

## 6. FK 的验证方法

- 零位测试：设 q=0，观察末端是否在预期方向；
- 单关节测试：只改变一个 q；
- 极限测试：关节到上下限；
- 数值与可视化：画连杆和末端；
- 与仿真器/机器人 SDK 对比；
- 对 T 检查 R 合法性和位置连续性。

## 7. 在 AI 与具身智能中的使用

FK 出现在：

- 机器人状态预处理；
- imitation learning 的末端 pose 标签；
- 视觉目标与关节目标转换；
- differentiable simulation；
- 运动规划与碰撞检查；
- World Model 的机器人状态表示；
- 视觉策略的 proprioception 特征。

典型流程：

```text
q_t
↓ FK
end-effector pose_t
↓ 与视觉/目标融合
policy / planner
↓
q_{t+1} 或 action
```

## 8. 常见错误与 Debug

### 8.1 角度单位错误

FK 中 sin/cos 通常需要弧度。输入度数会产生看似连续但位置完全错误的结果。

### 8.2 零位和轴方向不一致

机器人厂家、URDF、仿真器和自定义代码可能使用不同关节零位与轴方向。单关节测试最有效。

### 8.3 link/joint frame 混淆

固定 transform、joint transform 和末端工具偏置要区分。少一个 tool transform 就会有固定位置/姿态误差。

### 8.4 乘法顺序错误

沿链从 base 到 ee 逐步检查中间 frame，不要只对最终矩阵做猜测。

### 8.5 batch 维和关节维混淆

q 可能是 [B,n_joints]，每个 batch 都要输出对应 pose。检查广播和返回 shape。

## 9. 与其他笔记的关系

前置：[[坐标系与变换]]、[[旋转表示]]、[[线性代数]]、[[PyTorch]]。

后续：[[Inverse_Kinematics]]、[[Jacobian]]、[[Dynamics]]、[[Control]]、[[Manipulation]]。

本篇不展开从 pose 反求 q，见 [[Inverse_Kinematics]]。

## 10. 知识地图

```text
joint q
↓
joint axis / link geometry
↓
local transforms
↓
transform chain
↓
T_base←ee(q)
↓
position + orientation
↓
planning / control / learning
```

## 11. 快速查询表

| 检查项 | 重点 |
|---|---|
| q | 关节顺序、单位、零位 |
| link | 长度和固定偏置 |
| axis | 方向和 frame |
| chain | base 到 ee 顺序 |
| output | pose frame 和 shape |
| validate | 单关节、零位、可视化 |

## 12. 练习题（题目下附参考答案）

1. FK 回答什么问题？

**参考答案：** 已知关节变量，计算末端在参考坐标系中的位置和姿态。

2. joint 和 link 的区别是什么？

**参考答案：** joint 允许相对运动，link 是连接关节的刚体部分。

3. revolute joint 的变量通常是什么？

**参考答案：** 旋转角度。

4. prismatic joint 的变量通常是什么？

**参考答案：** 沿轴平移的长度。

5. 为什么第二连杆方向是 q1+q2？

**参考答案：** 第二关节角是相对第一连杆的旋转，绝对方向要累加前面的旋转。

6. FK 输出通常包含什么？

**参考答案：** 末端位置和姿态，即一个 pose/齐次变换。

7. DH 参数的作用是什么？

**参考答案：** 用统一规则描述相邻 frame 和连杆几何关系。

8. FK 中角度通常用什么单位？

**参考答案：** 弧度，除非接口明确要求角度。

9. 零位测试有什么作用？

**参考答案：** 验证轴方向、零位偏置、连杆长度和 frame 定义。

10. 只改变一个关节能检查什么？

**参考答案：** 检查该关节的轴方向、旋转方向和对末端的影响。

11. tool transform 漏掉会怎样？

**参考答案：** 末端位置或姿态出现固定偏差。

12. q=[B,n] 的 FK 输出应如何理解？

**参考答案：** 每个 batch 样本有 n 个关节，并得到对应的末端 pose。

13. FK 与 IK 的区别是什么？

**参考答案：** FK 从 q 求 pose，IK 从目标 pose 求 q。

14. 为什么 FK 可用于 imitation learning？

**参考答案：** 可把关节轨迹转换成末端位置/姿态标签或动作特征。

15. 为什么 FK 可用于碰撞检查？

**参考答案：** 它提供各连杆和末端的空间位置，供几何检查使用。

16. 位置正确但姿态错误应检查什么？

**参考答案：** 旋转表示、末端工具偏置、轴方向和 frame 变换。

17. FK 结果跳变可能来自什么？

**参考答案：** 角度单位/unwrap、关节顺序、四元数转换、frame 或数据时间跳变。

18. 如何检查 batch 广播错误？

**参考答案：** 打印 q、每级 transform 和输出 shape，并用单样本对比。

19. 仿真和真实 FK 不一致怎么查？

**参考答案：** 比较 URDF/DH、关节零位、轴、单位、base/ee frame 和工具偏置。

20. 读 FK 代码最先看什么？

**参考答案：** 关节顺序、轴、单位、局部变换顺序、输出 frame 和 shape。

## 13. 掌握度检查（附答案）

1. FK 的输入输出是什么？

**答案：** 输入关节变量 q，输出末端或各 link 的 pose。

2. kinematic chain 是什么？

**答案：** 从 base 到末端按关节和连杆连接形成的变换链。

3. revolute 与 prismatic 差别是什么？

**答案：** 一个绕轴转，一个沿轴移。

4. FK 为什么是变换相乘？

**答案：** 每个关节/连杆提供局部 frame 变换，连续相乘得到全局 pose。

5. DH 参数描述什么？

**答案：** 相邻关节坐标系和连杆几何关系。

6. 为什么单位影响 FK？

**答案：** 长度和角度进入 sin/cos 与平移，单位错会直接改变位置。

7. 零位测试检查什么？

**答案：** 结构几何、关节零位、轴方向和固定偏置。

8. FK 如何支持控制？

**答案：** 提供当前末端 pose，与目标比较形成误差。

9. FK 如何支持视觉策略？

**答案：** 将 proprioception 关节状态转换为空间末端特征。

10. FK 如何支持规划？

**答案：** 把候选 q 转为末端和连杆位置，检查目标与碰撞。

11. tool frame 为什么重要？

**答案：** 真正执行任务的是工具尖端，不一定是最后一个 link 原点。

12. FK 与 Jacobian 的关系是什么？

**答案：** Jacobian 是 FK 对关节变量的局部导数，描述速度和小位姿变化关系。

13. 何时查 [[坐标系与变换]]？

**答案：** 需要确认各级 frame 和 T 方向时。

14. 何时查 [[Inverse_Kinematics]]？

**答案：** 已知目标 pose，需要求关节角时。

15. 读 FK research code 最先确认哪五项？

**答案：** q 顺序、单位/零位、axis、chain、输出 pose frame。
