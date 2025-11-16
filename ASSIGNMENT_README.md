# 编程作业：从零实现MDP与贝尔曼方程验证

**基于《强化学习的数学基础》第1章和第2章**

---

## 📚 作业背景

本作业旨在通过动手编程，深刻理解强化学习的两个核心章节：

- **第1章：基本概念** - MDP（马尔可夫决策过程）、状态、动作、策略、奖励、回报
- **第2章：状态价值与贝尔曼方程** - 状态价值函数的定义、贝尔曼方程的推导与验证

## 🎯 核心学习目标

通过完成本作业，你将：

1. **理解MDP的数学结构**：从第一性原理出发，实现状态空间、动作空间、状态转移概率、奖励函数
2. **掌握状态价值函数**：理解状态价值函数 v_π(s) 的数学定义和物理意义
3. **验证贝尔曼方程**：通过代码实现验证贝尔曼方程的正确性，理解"价值函数满足的递归关系"
4. **掌握两种求解方法**：迭代法和矩阵形式求解，理解它们的等价性

## 📋 作业要求

### 必须完成的部分（核心逻辑）

你需要实现以下函数中的 `# TODO` 部分：

1. **MDP基本组件**（`GridWorldMDP`类）：
   - `state_to_index()` - 状态坐标到索引的转换
   - `index_to_state()` - 索引到状态坐标的转换
   - `is_valid_state()` - 状态有效性检查
   - `get_reward()` - 奖励函数
   - `get_transition_prob()` - 状态转移概率
   - `get_next_state()` - 状态转移函数

2. **策略定义**：
   - `create_uniform_policy()` - 创建均匀随机策略

3. **状态价值计算**（核心）：
   - `compute_state_value_iterative()` - 使用迭代法求解贝尔曼方程
   - `compute_state_value_matrix_form()` - 使用矩阵形式求解（可选，用于验证）

4. **贝尔曼方程验证**（核心）：
   - `verify_bellman_equation()` - 验证计算出的状态价值是否满足贝尔曼方程

### 成功标准

作业完成的标志是：

✅ **状态价值函数满足贝尔曼方程，最大误差 < 1e-4**

运行 `python assignment_ch1_ch2.py` 后，应该看到：
```
✓ 成功！状态价值函数满足贝尔曼方程（误差 < 1e-4）
✓ 作业完成标准：已达成
```

## 🔧 环境设置

1. **激活虚拟环境**（如果还没有）：
   ```powershell
   .\env\Scripts\Activate.ps1
   ```

2. **安装依赖**：
   ```powershell
   pip install numpy matplotlib
   ```

3. **运行作业**：
   ```powershell
   python assignment_ch1_ch2.py
   ```

## 📖 关键数学公式

### 贝尔曼方程（第2章核心）

对于给定的策略 π，状态价值函数 v_π(s) 满足：

```
v_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ * v_π(s')]
```

其中：
- `π(a|s)` 是策略：在状态s下选择动作a的概率
- `P(s'|s,a)` 是状态转移概率：在状态s执行动作a后到达状态s'的概率
- `R(s,a,s')` 是奖励函数：从状态s执行动作a到达状态s'获得的奖励
- `γ` 是折扣因子：未来奖励的折扣率

### 迭代求解方法

从初始值 v₀(s) = 0 开始，迭代更新：

```
v_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ * v_k(s')]
```

直到收敛：|v_{k+1} - v_k| < tolerance

### 矩阵形式求解

```
v_π = (I - γP_π)^(-1) * r_π
```

其中：
- `P_π` 是考虑策略后的状态转移概率矩阵
- `r_π` 是考虑策略后的奖励向量

## 💡 实现提示

### 1. 状态索引转换

在Grid World中，状态用二维坐标 `(x, y)` 表示，但在计算中需要转换为一维索引：

```python
# 状态 (x, y) -> 索引 i
index = y * width + x

# 索引 i -> 状态 (x, y)
x = index % width
y = index // width
```

### 2. 状态转移概率

在确定性环境中（本作业），状态转移是确定的：
- 如果 `next_state == get_next_state(state, action)`，则 `P(next_state|state, action) = 1.0`
- 否则 `P(next_state|state, action) = 0.0`

### 3. 迭代法实现要点

对于每个状态 s，计算：

```python
new_value = 0
for action_idx, action in enumerate(actions):
    action_prob = policy[state_idx, action_idx]  # π(a|s)
    
    next_state = get_next_state(state, action)
    next_state_idx = state_to_index(next_state)
    reward = get_reward(state, action, next_state)
    trans_prob = get_transition_prob(state, action, next_state)  # P(s'|s,a)
    
    new_value += action_prob * trans_prob * (reward + gamma * old_values[next_state_idx])
```

### 4. 验证贝尔曼方程

对于每个状态 s，计算贝尔曼方程右侧的值，与左侧的 `state_values[s]` 比较：

```python
# 左侧：v_π(s)
left_side = state_values[state_idx]

# 右侧：Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ * v_π(s')]
right_side = 0
for action_idx, action in enumerate(actions):
    # ... 计算期望
    right_side += ...

error = abs(left_side - right_side)
```

## 🐛 调试建议

1. **先实现基础函数**：确保 `state_to_index`、`get_next_state` 等基础函数正确
2. **测试小例子**：可以用一个2x2的网格先测试
3. **打印中间结果**：在迭代过程中打印状态价值的变化
4. **检查边界情况**：确保障碍物、边界状态处理正确

## 📝 作业提交

完成作业后，你应该能够：

1. ✅ 成功运行代码，看到可视化结果
2. ✅ 验证通过：贝尔曼方程误差 < 1e-4
3. ✅ 理解每个函数的作用和数学含义

## 🔗 参考资源

- 赵世钰老师《强化学习的数学基础》第1章、第2章
- Grid World代码示例：`Book-Mathematical-Foundation-of-Reinforcement-Learning/Code for grid world/`
- 课程视频：[Bilibili](https://space.bilibili.com/2044042934) 或 [YouTube](https://www.youtube.com/channel/UCztGtS5YYiNv8x3pj9hLVgg/playlists)

---

**祝你学习愉快！记住：理解数学原理比完成代码更重要。** 🚀

