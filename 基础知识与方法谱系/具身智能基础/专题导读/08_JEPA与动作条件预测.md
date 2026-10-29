# JEPA 与动作条件预测

JEPA 在表征空间构造预测任务。它与世界模型相关，但不能仅凭“预测 latent”就推断模型已经具备动作选择能力。

## 从 I-JEPA 的任务看起

I-JEPA 从图像上下文预测同一图像中目标块的表示。可用下面的示意式理解其分支关系；具体掩码、归一化和损失形式应以论文为准：

$$
\hat z_y=p_\theta(e_\theta(x_{context}),m_y),\qquad
z_y=\operatorname{sg}(e_{\bar\theta}(x)_{target}).
$$

这里 $m_y$ 提供目标位置信息，$\operatorname{sg}$ 表示停止该目标路径的梯度。目标编码器采用相应的移动平均更新机制。仅有 stop-gradient 并不能普遍保证所有自监督方法避免坍塌。[I-JEPA 原论文](https://arxiv.org/abs/2301.08243)

## 视频预测与动作预测分开看

从视频上下文预测未来表示，可能学习视觉时序规律；若要比较候选机器人动作，还需要能表达动作后果的动态模型。V-JEPA 2 的论文区分视频预训练与动作条件后训练 V-JEPA 2-AC，并进一步讨论规划。[V-JEPA 2，§2–4](https://arxiv.org/html/2506.09985v1)

## 梯度范围的概念例子

若代码是 `trajectory = readout(predicted_latent.detach())`，加在 `trajectory` 上的几何损失不会沿这条边更新 predictor。它仍可更新 readout；predictor 也可能被另一项独立损失更新。

因此，“轨迹更一致”与“潜空间学到物理约束”需要分别证明。定位梯度范围时应检查 detach、冻结参数、优化器参数组及所有损失路径，不能只看模块名称。

## 阅读清单

- 预测的是同一图像的遮挡区域，还是未来时间段？
- 模型是否接收动作，动作对应几个视频帧？
- target 是否以未预期的路径泄漏给 predictor？
- 报告的是表示指标、轨迹指标，还是闭环控制结果？

下一篇：[规划与 MPC](09_规划与MPC.md)。
