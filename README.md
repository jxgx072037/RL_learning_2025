# 强化学习作业：MDP与贝尔曼方程验证

基于《强化学习的数学基础》第1章和第2章的编程实践项目。

## 📖 项目简介

本项目通过实现一个简单的 Grid World 环境，从零开始构建马尔可夫决策过程（MDP），并验证贝尔曼方程的正确性。这是一个学习强化学习基础概念的教学项目。

## 🎯 核心内容

- **第1章概念**：MDP、状态、动作、策略、奖励、回报
- **第2章概念**：状态价值函数、贝尔曼方程
- **实现功能**：
  - Grid World MDP 环境
  - 状态价值函数计算（迭代法和矩阵形式）
  - 贝尔曼方程验证

## 🚀 快速开始

### 环境要求

- Python 3.8+
- NumPy
- Matplotlib

### 安装依赖

```bash
pip install numpy matplotlib
```

### 运行作业

```bash
python assignment_ch1_ch2.py
```

成功运行后，你会看到：
- 状态价值函数的计算结果
- 贝尔曼方程验证结果
- 状态价值函数的可视化图表

## 📁 项目结构

```
.
├── assignment_ch1_ch2.py          # 主程序：MDP实现和贝尔曼方程验证
├── ASSIGNMENT_README.md            # 详细作业说明
├── learning_guide.md               # 学习指南
├── test_pdf_reader.py              # PDF读取工具（辅助）
├── Book-Mathematical-Foundation-of-Reinforcement-Learning/  # 学习资料
│   ├── 3 - Chapter 1 Basic Concepts.pdf
│   ├── 3 - Chapter 2 State Values and Bellman Equation.pdf
│   └── Code for grid world/       # 参考代码
└── README.md                       # 本文件
```

## 🔑 关键概念

### 贝尔曼方程

对于给定的策略 π，状态价值函数 v_π(s) 满足：

```
v_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ * v_π(s')]
```

### 求解方法

1. **迭代法**：通过迭代更新状态价值直到收敛
2. **矩阵形式**：直接求解 `v_π = (I - γP_π)^(-1) * r_π`

## ✅ 成功标准

作业完成的标志是：
- ✅ 状态价值函数满足贝尔曼方程
- ✅ 最大误差 < 1e-4

## 📚 参考资料

- 《强化学习的数学基础》- 赵世钰
- [课程视频](https://space.bilibili.com/2044042934)

## 📝 学习笔记

详细的学习指南和实现说明请参考：
- `ASSIGNMENT_README.md` - 完整作业要求
- `learning_guide.md` - 学习笔记

## 🤝 贡献

这是一个学习项目，欢迎提出问题和改进建议！

---

**注意**：本项目仅用于学习目的，遵循原教材的教学内容。

