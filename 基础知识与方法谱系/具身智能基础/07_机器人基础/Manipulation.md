# Manipulation：让机器人与物体安全地交互

机器人操作（Robotic Manipulation）研究机器人如何感知、接近、抓取、移动、放置和与物体接触。它把视觉、坐标系、运动学、动力学、控制和学习策略组合成一个闭环任务。

本文负责操作任务的整体视角、抓取、接触、动作表示、失败分析和 research code 阅读；具体几何见 [[坐标系与变换]]、[[Forward_Kinematics]]、[[Inverse_Kinematics]]，控制见 [[Control]]，视觉见 [[视觉表征]]。

## 0. 学习目标

学完以后能够：

- 描述 manipulation pipeline 的完整阶段；
- 区分 reaching、grasping、transport、placing 和 contact；
- 理解抓取需要的视觉、几何和控制信息；
- 读懂 observation、action、goal 和 success 的数据结构；
- 分析 sim-to-real、接触、延迟和闭环失败；
- 把机器人基础连接到 imitation learning、RL 和 World Model。

## 1. 为什么操作任务难？

抓取不是“看到物体就输出一个坐标”。机器人需要：

```text
感知物体
↓
估计位置/姿态/几何
↓
选择抓取点与接近方向
↓
规划无碰撞路径
↓
IK/控制跟踪
↓
接触与闭合
↓
验证抓取成功
↓
搬运/放置/恢复
```

每个环节都有不确定性，且动作会改变物体和后续观测。

## 2. 核心概念

### 2.1 Reaching 与 Grasping

- reaching：末端移动到目标附近；
- grasping：通过夹爪/吸盘等建立稳定约束；
- transport：保持抓取并移动；
- placing：在目标区域释放；
- recovery：失败后重新定位或尝试。

### 2.2 Grasp Pose

抓取 pose 包含位置和方向。抓取点还要考虑：

- 夹爪开口和方向；
- 物体形状与接触面；
- 接近路径；
- 碰撞；
- 摩擦和负载；
- 释放姿态。

### 2.3 Contact

接触会引入力、摩擦、碰撞和约束。位置误差小不代表接触稳定；可能发生滑动、卡住、撞击或物体变形。

### 2.4 Observation、Action、Goal

一个 manipulation policy 的 observation 可能包含：

- RGB/depth/segmentation；
- 关节位置速度；
- 末端 pose；
- 夹爪状态；
- 目标 pose；
- 历史帧。

action 可能是关节速度、末端 delta pose、夹爪开合或 action chunk。

### 2.5 Success 与 Failure

不能只看单步动作误差。任务成功可能要求：

- 目标被正确抓取；
- 物体没有掉落；
- 放置位置满足阈值；
- 没有碰撞；
- 在时间限制内完成。

## 3. Manipulation 的整体流程

```text
reset scene
↓
observe RGB/depth/state
↓
detect/represent object
↓
transform target to robot frame
↓
plan grasp/reach
↓
IK + controller
↓
contact/grasp
↓
verify force/vision/gripper state
↓
transport/place
↓
success or recovery
```

## 4. 从零读一个操作策略接口

```python
obs = {
    "rgb": rgb,
    "joint_pos": q,
    "eef_pose": eef_pose,
    "gripper": gripper_state,
    "goal": goal_pose,
}
action = policy(obs)
next_obs, reward, terminated, truncated, info = env.step(action)
```

要确认：

- obs 每个字段的 frame、shape、单位和时间；
- action 是目标还是增量；
- policy 输出由谁限幅；
- reward/success 是否在真正完成动作后计算；
- next_obs 是否由 action 后时刻产生。

## 5. 视觉与几何链路

```text
图像像素/深度
↓
相机 frame 目标
↓ 外参
机器人 base frame
↓
目标 grasp pose
↓ IK
joint target
↓
controller
↓
接触与抓取
```

任何 frame、单位或时间错误都会传到最终动作。视觉表征是否保留空间信息见 [[视觉表征]]。

## 6. 学习方法

### 6.1 Imitation Learning

从专家操作轨迹学习 observation → action，数据效率高但有分布偏移，见 [[Imitation_Learning]]。

### 6.2 Reinforcement Learning

根据成功、距离、碰撞等 reward 通过交互优化策略，见 [[MDP]]、[[Actor_Critic]]。

### 6.3 Model-Based

学习动作后的状态/latent，再规划未来操作，见 [[Model_Based_RL]]。

### 6.4 World Model

在 latent 中预测物体、机器人和动作的未来，用于规划和想象训练。

## 7. Sim-to-Real

仿真操作策略迁移到真实机器人时，要考虑：

- 相机内外参和图像分布；
- 摩擦、质量、接触和柔性；
- 执行器延迟与控制频率；
- 物体形状、摆放和随机性；
- 传感器噪声；
- 安全限位与恢复。

随机化可能帮助泛化，但必须保持任务语义。

## 8. 常见错误与 Debug

### 8.1 目标 pose frame 错

视觉目标没正确转换到 base/control frame，机器人会稳定地去错误位置。

### 8.2 抓取姿态只有位置没有方向

夹爪方向不合适会碰撞、夹不住或从侧面滑落。

### 8.3 action 是 delta 还是 absolute 混淆

策略输出小增量却被当成绝对位置，动作会跳变；反之则动作几乎不动。

### 8.4 成功标签过早

夹爪刚闭合就判成功，实际物体可能掉落。应结合后续视觉、力或状态验证。

### 8.5 只测 open-loop

离线动作看起来合理，闭环执行后误差累积。必须在仿真/真实环境中 rollout。

### 8.6 失败没有恢复策略

遇到遮挡、滑落或 IK 无解时，系统应重新观测、放弃或退回安全状态。

## 9. 与其他笔记的关系

前置：[[图像基础]]、[[视觉表征]]、[[坐标系与变换]]、[[旋转表示]]、[[Forward_Kinematics]]、[[Inverse_Kinematics]]、[[Control]]。

相关：[[Imitation_Learning]]、[[MDP]]、[[Actor_Critic]]、[[Model_Based_RL]]、[[Offline_RL]]、[[Transformer]]。

本篇不展开某一种抓取算法或具体机械臂型号，重点建立跨模块任务视角。

## 10. 知识地图

```text
perception
↓
geometry
↓
planning
↓
IK / control
↓
contact
↓
verification
├─ success
└─ recovery
```

## 11. 快速查询表

| 阶段 | 核心检查 |
|---|---|
| perception | 目标、深度、时间 |
| geometry | frame、单位、pose |
| planning | 路径、碰撞、可达 |
| control | 频率、延迟、限幅 |
| contact | 力、摩擦、夹爪 |
| success | 物体状态和目标条件 |
| recovery | 重新观察和安全回退 |

## 12. 练习题（题目下附参考答案）

1. manipulation 包含哪些典型阶段？

**参考答案：** 感知、定位、接近、抓取、搬运、放置、验证和失败恢复。

2. reaching 和 grasping 的区别是什么？

**参考答案：** reaching 是到达目标附近，grasping 是建立稳定接触并约束物体。

3. grasp pose 为什么包含方向？

**参考答案：** 夹爪接近方向和姿态决定碰撞、接触和稳定性。

4. contact 会带来什么复杂性？

**参考答案：** 力、摩擦、碰撞、约束和物体运动变化。

5. action 可能是什么形式？

**参考答案：** 关节位置/速度/力矩、末端 delta pose、夹爪命令或 action chunk。

6. 为什么目标要从 camera frame 转到 robot frame？

**参考答案：** 控制和 IK 需要在机器人使用的参考系中解释目标。

7. 为什么不能只看一步动作误差？

**参考答案：** 任务成功依赖连续闭环、接触和最终物体状态。

8. 仿真与真实差异有哪些？

**参考答案：** 视觉、摩擦、接触、质量、延迟、噪声和执行器限制。

9. action absolute/delta 混淆会怎样？

**参考答案：** 命令可能跳变或几乎不动，导致运动异常。

10. 为什么成功标签不能太早？

**参考答案：** 夹爪闭合并不代表物体被稳定抓住或已放置成功。

11. 失败恢复有什么作用？

**参考答案：** 处理遮挡、滑落、碰撞或不可达，避免系统持续执行错误动作。

12. 视觉表征需要保留什么？

**参考答案：** 目标身份、位置、姿态、距离和动作相关信息。

13. IK 在操作链中位于哪里？

**参考答案：** 目标 grasp pose 确定后，把它转换为关节配置或轨迹。

14. 控制器在操作链中做什么？

**参考答案：** 跟踪关节/末端目标并处理反馈、限幅和安全。

15. RL reward 可以包括什么？

**参考答案：** 成功、距离进展、碰撞惩罚、时间成本和稳定抓取奖励。

16. 模仿学习的主要风险是什么？

**参考答案：** 训练状态分布与部署状态分布不同，误差可能累积。

17. 如何调试一次抓取失败？

**参考答案：** 回放视觉、frame transform、grasp pose、IK error、控制命令、接触力和最终状态。

18. 为什么要记录时间戳？

**参考答案：** 对齐图像、状态、动作和接触事件，避免因果错位。

19. World Model 如何服务 manipulation？

**参考答案：** 预测动作后的物体/机器人 latent，帮助规划候选操作和评估结果。

20. 读 manipulation repository 最先看什么？

**参考答案：** observation/action schema、frame、控制频率、成功定义、数据来源和 rollout 流程。

## 13. 掌握度检查（附答案）

1. 操作任务为什么是闭环问题？

**答案：** 动作会改变机器人、物体和未来观测，后续决策依赖反馈。

2. reaching 与 grasping 的区别是什么？

**答案：** reaching 到目标附近，grasping 建立稳定物体约束。

3. 抓取 pose 包含什么？

**答案：** 位置和方向，还隐含接近方向、夹爪和物体几何约束。

4. 为什么接触使问题更难？

**答案：** 接触引入力、摩擦、碰撞、约束和不连续变化。

5. 操作 observation 可能包含什么？

**答案：** RGB/depth、关节状态、末端 pose、夹爪、目标和历史。

6. action 为什么必须有单位和 frame？

**答案：** 同一数字在不同坐标和单位下代表完全不同的运动。

7. 成功评估为什么要任务级？

**答案：** 单步动作或离线 loss 不能说明物体最终是否正确完成任务。

8. Sim-to-real 的主要风险是什么？

**答案：** 仿真与真实的视觉、动力学、接触、延迟和噪声分布不同。

9. 为什么要有 recovery？

**答案：** 真实环境中观测和执行不确定，失败后需要安全重新规划。

10. 模仿学习如何进入操作任务？

**答案：** 从专家轨迹学习观测到动作，再通过闭环执行验证。

11. RL 如何进入操作任务？

**答案：** 用成功、距离、碰撞和时间等 reward 通过交互优化策略。

12. World Model 如何进入操作任务？

**答案：** 预测动作对未来物体/机器人状态的影响，用于规划和选择动作。

13. 何时查 [[坐标系与变换]]？

**答案：** 视觉目标、末端和机器人 base 之间 frame 不一致时。

14. 何时查 [[Control]]？

**答案：** 需要分析频率、延迟、反馈、限幅和稳定性时。

15. 读 manipulation research code 最先确认哪五项？

**答案：** observation/action、时间、frame/单位、成功定义、闭环与恢复流程。
