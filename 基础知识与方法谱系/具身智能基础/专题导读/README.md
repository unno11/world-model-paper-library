# 具身智能基础专题导读

按“坐标变换 → 运动学 → 控制 → 世界模型 → 规划”串联基础概念。每篇包含约定、推导或例子、适用边界和原始资料链接。

| 阅读顺序 |
| --- |
| [坐标系与刚体变换](01_%E5%9D%90%E6%A0%87%E7%B3%BB%E4%B8%8E%E5%88%9A%E4%BD%93%E5%8F%98%E6%8D%A2.md) |
| [旋转表示与姿态误差](02_%E6%97%8B%E8%BD%AC%E8%A1%A8%E7%A4%BA.md) |
| [正向运动学：从关节到末端](03_%E6%AD%A3%E5%90%91%E8%BF%90%E5%8A%A8%E5%AD%A6.md) |
| [逆运动学与雅可比](04_%E9%80%86%E8%BF%90%E5%8A%A8%E5%AD%A6%E4%B8%8E%E9%9B%85%E5%8F%AF%E6%AF%94.md) |
| [动力学与反馈控制](05_%E5%8A%A8%E5%8A%9B%E5%AD%A6%E4%B8%8E%E5%8F%8D%E9%A6%88%E6%8E%A7%E5%88%B6.md) |
| [状态、观测与控制表征](06_%E7%8A%B6%E6%80%81%E8%A7%82%E6%B5%8B%E4%B8%8E%E6%8E%A7%E5%88%B6%E8%A1%A8%E5%BE%81.md) |
| [世界模型：预测如何服务决策](07_%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B.md) |
| [JEPA 与动作条件预测](08_JEPA%E4%B8%8E%E5%8A%A8%E4%BD%9C%E6%9D%A1%E4%BB%B6%E9%A2%84%E6%B5%8B.md) |
| [规划与模型预测控制](09_%E8%A7%84%E5%88%92%E4%B8%8EMPC.md) |
| [潜空间规划：从表征距离到实际目标](10_%E6%BD%9C%E7%A9%BA%E9%97%B4%E8%A7%84%E5%88%92.md) |

## 数值例子

[5 个 NumPy 示例](examples/README.md)覆盖坐标往返、FK/雅可比、一阶反馈、线性动态模型和 MPC。例子用于解释概念，不属于论文复现成绩。

## 连接论文

基础概念与具体论文分开维护，论文方法、版本和发表信息见对应条目。

- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](../../../01_%E6%BD%9C%E7%A9%BA%E9%97%B4%E8%A7%84%E5%88%92%E4%B8%8E%E8%A1%A8%E7%A4%BA/ICML/2411.04983%20DINO-WM%20%20World%20Models%20on%20Pre-trained%20Visual%20Features%20enable%20Zero-shot%20Planning.md)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](../../../01_%E6%BD%9C%E7%A9%BA%E9%97%B4%E8%A7%84%E5%88%92%E4%B8%8E%E8%A1%A8%E7%A4%BA/%E9%A2%84%E5%8D%B0%E6%9C%AC/2506.09985%20V-JEPA%202%20%20Self-Supervised%20Video%20Models%20Enable%20Understanding%2C%20Prediction%20and%20Planning.md)
- [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](../../../01_%E6%BD%9C%E7%A9%BA%E9%97%B4%E8%A7%84%E5%88%92%E4%B8%8E%E8%A1%A8%E7%A4%BA/%E9%A2%84%E5%8D%B0%E6%9C%AC/2603.14482%20V-JEPA%202.1%20%20Unlocking%20Dense%20Features%20in%20Video%20Self-Supervised%20Learning.md)

[资料索引](资料索引.md) · [返回基础知识总目录](../README.md)
