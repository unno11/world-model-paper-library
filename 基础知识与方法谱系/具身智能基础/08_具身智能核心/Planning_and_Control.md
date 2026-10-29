# Planning and Control：从“想做什么”到“安全地做出来”

Planning and Control 描述具身系统如何把任务目标转成行动：规划器选择未来路径或子目标，控制器在反馈下跟踪并修正。

本文负责 planner、controller、层级结构、模型预测、安全和接口；机器人控制基础见 [[Control]]，Latent Planning 见 [[Latent_Planning]]，World Model 见 [[World_Model]]。

## 0. 学习目标

学完以后能够：

- 区分 task planning、motion planning、policy 和 controller；
- 理解 open-loop plan 与 receding-horizon control；
- 追踪 goal、trajectory、action 和 feedback；
- 看懂视觉/latent planner 接到机器人控制的代码；
- 分析约束、碰撞、延迟和不确定性；
- 排查规划正确但执行失败的接口问题。

## 1. 为什么需要分层？

高层任务“把杯子放到桌上”与低层“电机在 1 ms 后输出多少力矩”不是同一个问题。

```text
任务目标
↓ task planner
技能/子目标
↓ motion planner
空间轨迹
↓ IK / controller
关节命令
↓
机器人与环境
↓ feedback
└───────────────
```

分层使每个模块处理合适的时间尺度和约束。

## 2. 核心概念

### 2.1 Task Planning

选择技能序列、物体和高层步骤，如 reach → grasp → lift → place。

### 2.2 Motion Planning

在几何和碰撞约束下规划连续空间轨迹。

### 2.3 Policy

根据观测直接输出 action 或子目标，可能替代部分规划器。

### 2.4 Controller

高频跟踪目标，处理反馈、动力学、限幅和安全。

### 2.5 Trajectory

轨迹是按时间排列的状态、pose、速度或动作目标。规划轨迹必须满足机器人能力和任务约束。

### 2.6 Receding Horizon

只执行当前计划的前几步，然后重新观察和规划，降低模型和感知误差影响。

## 3. Planning 与 Control 的整体流程

```text
goal/observation
↓
task or latent planner
↓
candidate trajectory
↓
collision/IK/dynamics filter
↓
controller
↓
action
↓
sensor feedback
↓
replan
```

## 4. 从零读接口代码

```python
goal = get_goal(obs)
plan = planner.plan(obs, goal)

for waypoint in plan:
    command = controller.compute(
        target=waypoint,
        state=robot_state,
    )
    safe_command = safety_filter(command)
    robot.send(safe_command)
```

检查 waypoint 是 pose、velocity 还是 joint target；controller 期待的 frame、unit 和频率必须一致。

## 5. 约束与安全

规划应考虑：

- joint limits；
- workspace；
- collision；
- velocity/acceleration；
- force/torque；
- time；
- grasp stability；
- human/robot safety。

安全 filter 可以拒绝、限幅或替换风险动作，但不能成为“出了问题再补救”的唯一机制。

## 6. 学习策略与规划的组合

常见组合：

- policy 提议动作，planner 筛选；
- planner 给 subgoal，policy 执行；
- World Model 预测，planner 选择；
- policy warm-start，MPC 修正；
- controller 负责最后跟踪。

组合代码中要确认梯度是否跨模块、训练和部署是否使用同一接口。

## 7. 在具身智能中的使用

```text
视觉/语言/状态
↓
goal or latent representation
↓
planner / policy
↓
pose or action chunk
↓
IK
↓
low-level controller
↓
contact and success monitor
```

## 8. 常见错误与 Debug

### 8.1 planner 输出不可执行

检查 IK、碰撞、速度、姿态和 workspace。

### 8.2 controller 跟不上

检查目标跳变、频率、延迟、增益、动力学和执行器饱和。

### 8.3 计划没有重新规划

开环计划遇到感知/动力学偏差后继续执行，误差会累积。

### 8.4 planner 与 controller frame 不一致

计划在 world，控制器按 base 解读，动作方向错误。

### 8.5 目标层级不一致

高层给 object goal，低层需要 eef pose 或 joint target，中间转换缺失。

## 9. 与其他笔记的关系

前置：[[Control]]、[[Inverse_Kinematics]]、[[Dynamics]]、[[MDP]]、[[World_Model]]。

相关：[[Latent_Planning]]、[[VLA]]、[[Robot_Learning]]、[[Manipulation]]、[[Diffusion_Policy]]。

## 10. 知识地图

```text
task goal
↓
task/latent planner
↓
motion trajectory
↓
IK / feasibility
↓
controller
↓
safety filter
↓
robot
↓
feedback/replan
```

## 11. 快速查询表

| 接口 | 先问什么 |
|---|---|
| planner output | goal/pose/joint/action |
| trajectory | 时间、速度、加速度 |
| IK | frame、可达、限位 |
| controller | 输入单位和频率 |
| safety | 限幅、拒绝、急停 |
| replan | 何时使用新观测 |

## 12. 练习题（题目下附参考答案）

1. Planning 与 Control 的区别是什么？

**参考答案：** Planning 选择未来路径/目标，Control 通过反馈跟踪目标并处理执行误差。

2. task planning 做什么？

**参考答案：** 选择高层技能和任务步骤。

3. motion planning 做什么？

**参考答案：** 在几何和碰撞约束下生成可行空间轨迹。

4. controller 做什么？

**参考答案：** 高频根据误差产生执行器命令。

5. receding horizon 的流程是什么？

**参考答案：** 规划一段、执行前几步、获得新观测、重新规划。

6. 为什么需要 IK filter？

**参考答案：** 过滤末端目标不可达或超出关节限制的计划。

7. 为什么需要 collision check？

**参考答案：** 数学轨迹可能穿过障碍或机器人自身。

8. policy 输出 action chunk 后谁负责执行？

**参考答案：** 通常由低层控制器按频率跟踪和限幅。

9. planner 与 controller frame 不一致会怎样？

**参考答案：** 计划方向会被错误解释，机器人执行偏离目标。

10. 计划正确但执行失败查什么？

**参考答案：** frame、单位、IK、控制频率、延迟、增益、饱和和动力学。

11. 何时需要 replan？

**参考答案：** 观测、目标、环境或模型预测发生变化时。

12. 高层目标与低层命令的差异是什么？

**参考答案：** 高层可能是物体目标或 pose，低层需要 joint/velocity/torque 命令。

13. World Model 如何帮助 planner？

**参考答案：** 在模型中预测候选动作的未来结果。

14. Diffusion Policy 与 planner 如何组合？

**参考答案：** policy 生成动作候选或轨迹，planner/安全层筛选和约束。

15. 为什么控制频率要高于规划频率？

**参考答案：** 低层需要更快反馈稳定运动，高层不必每个电机周期重新规划。

16. safety filter 能解决所有问题吗？

**参考答案：** 不能，它只能过滤部分风险，根本问题仍需数据、规划和控制修复。

17. 如何定义计划成功？

**参考答案：** 轨迹可行、无碰撞、目标达成且执行误差在容忍范围内。

18. 接触任务为什么不能只看位置？

**参考答案：** 还要控制力、姿态、摩擦和物体状态。

19. 研究代码中 planner 输出 shape 错怎么查？

**参考答案：** 明确 candidate、time、action、batch 维，并与 controller 接口对照。

20. 读 planning/control repository 最先看什么？

**参考答案：** 目标格式、规划输出、IK转换、控制频率、安全层和反馈循环。

## 13. 掌握度检查（附答案）

1. 为什么需要 planning/control 分层？

**答案：** 任务、几何和执行处于不同时间尺度，分层可降低复杂度并满足安全约束。

2. task planning 与 motion planning 区别是什么？

**答案：** 前者选技能/步骤，后者生成连续空间运动轨迹。

3. policy 与 planner 的差异是什么？

**答案：** policy 可直接反应生成动作，planner 通常显式比较未来或约束。

4. controller 的核心是什么？

**答案：** 反馈跟踪目标并修正误差。

5. receding horizon 为什么稳健？

**答案：** 通过不断获得真实反馈，限制长期错误计划的影响。

6. 为什么规划要通过 IK 和碰撞检查？

**答案：** 确认任务目标能被机器人实际、安全执行。

7. frame/单位错误为什么危险？

**答案：** 命令的方向和幅度会被错误解释，可能直接造成碰撞。

8. 为什么需要 safety filter？

**答案：** 限制策略和规划的越界动作，保护设备和人员。

9. planner 输出可是什么？

**答案：** 子目标、pose、joint trajectory、动作序列或 latent action proposal。

10. controller 输入可是什么？

**答案：** 目标位置/速度/力矩、当前状态和时间参数。

11. 为什么要 replan？

**答案：** 适应新观测、扰动、模型误差和目标变化。

12. 具身系统的最终评估是什么？

**答案：** 任务成功、执行安全、效率、泛化和闭环稳定性。

13. 何时查 [[Control]]？

**答案：** 需要理解 PID、频率、延迟和低层反馈时。

14. 何时查 [[Latent_Planning]]？

**答案：** 需要在 latent 中预测和搜索未来时。

15. 读 planning/control research code 最先确认哪五项？

**答案：** goal、trajectory/action、frame/unit、feasibility/safety、feedback/replan。
