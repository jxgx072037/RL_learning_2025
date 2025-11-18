"""
编程作业：从零实现MDP与贝尔曼方程验证
基于《强化学习的数学基础》第1章和第2章

第1章核心概念：MDP（马尔可夫决策过程）、状态、动作、策略、奖励、回报
第2章核心概念：状态价值函数、贝尔曼方程

作业目标：通过实现一个简单的Grid World环境，验证贝尔曼方程的正确性
"""

import numpy as np
from typing import Any, Tuple, List, Dict
import matplotlib.pyplot as plt


# ============================================================================
# 第一部分：定义MDP的基本组件（第1章内容）
# ============================================================================

class GridWorldMDP:
    """
    一个简单的Grid World MDP实现
    
    环境描述：
    - 网格大小：4x4
    - 起始状态：(0, 0)
    - 目标状态：(3, 3)
    - 障碍物：[(1, 1), (2, 2)]
    - 动作空间：上(0,-1), 右(1,0), 下(0,1), 左(-1,0)
    - 奖励：到达目标+10，撞墙/障碍物-1，普通步-0.1
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (4, 4),
                 start_state: Tuple[int, int] = (0, 0),
                 target_state: Tuple[int, int] = (3, 3),
                 obstacles: List[Tuple[int, int]] = None,
                 reward_target: float = 10.0,
                 reward_forbidden: float = -1.0,
                 reward_step: float = -0.1,
                 gamma: float = 0.9):
        """
        初始化Grid World MDP
        
        Args:
            grid_size: 网格大小 (width, height)
            start_state: 起始状态坐标
            target_state: 目标状态坐标
            obstacles: 障碍物坐标列表
            reward_target: 到达目标的奖励
            reward_forbidden: 撞墙/障碍物的奖励
            reward_step: 每步的奖励
            gamma: 折扣因子
        """
        self.grid_size = grid_size
        self.start_state = start_state
        self.target_state = target_state
        self.obstacles = obstacles if obstacles else [(1, 1), (2, 2)]
        self.reward_target = reward_target
        self.reward_forbidden = reward_forbidden
        self.reward_step = reward_step
        self.gamma = gamma
        
        # 动作空间：上、右、下、左
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        # 状态空间：所有可能的网格位置
        self.num_states = grid_size[0] * grid_size[1]
        
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """
        将二维状态坐标转换为一维索引
        
        TODO: 实现状态坐标到索引的转换
        提示：使用公式 index = y * width + x
        
        Args:
            state: (x, y) 坐标
            
        Returns:
            状态索引 (0 到 num_states-1)
        """
        # TODO: 在这里实现状态到索引的转换
        return state[0] * self.grid_size[1] + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """
        将一维索引转换为二维状态坐标
        
        TODO: 实现索引到状态坐标的转换
        提示：使用公式 x = index % width, y = index // width
        
        Args:
            index: 状态索引
            
        Returns:
            (x, y) 坐标
        """
        # TODO: 在这里实现索引到状态的转换
        return (index % self.grid_size[1], index // self.grid_size[1])
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """
        检查状态是否有效（在网格内且不是障碍物）
        
        TODO: 实现状态有效性检查
        提示：检查坐标是否在范围内，且不在障碍物列表中
        
        Args:
            state: (x, y) 坐标
            
        Returns:
            True如果状态有效，False否则
        """
        # TODO: 在这里实现状态有效性检查
        if state in self.obstacles:
            return False
        elif state[0] < 0 or state[0] >= self.grid_size[0] or state[1] < 0 or state[1] >= self.grid_size[1]:
            return False
        else:
            return True
    
    def get_reward(self, state: Tuple[int, int], action: Tuple[int, int], next_state: Tuple[int, int]) -> float:
        """
        获取状态转移的奖励
        
        TODO: 实现奖励函数
        提示：
        - 如果next_state是目标状态，返回reward_target
        - 如果next_state是障碍物或出界，返回reward_forbidden
        - 否则返回reward_step
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            
        Returns:
            奖励值
        """
        # TODO: 在这里实现奖励函数
        if next_state in self.obstacles:
            return self.reward_forbidden
        elif next_state == self.target_state:
            return self.reward_target
        elif next_state[0]<0 or next_state[0] >= self.grid_size[0] or next_state[1] <0 or next_state[1] >= self.grid_size[1]:
            return self.reward_forbidden
        else:
            return self.reward_step
    
    def get_transition_prob(self, state: Tuple[int, int], action: Tuple[int, int], next_state: Tuple[int, int]) -> float:
        """
        获取状态转移概率 P(s'|s,a)
        
        在这个确定性环境中，转移概率要么是1（如果next_state是执行action后的唯一可能结果），
        要么是0（如果next_state不可能从state通过action到达）
        
        TODO: 实现状态转移概率
        提示：
        1. 计算执行action后应该到达的状态
        2. 如果next_state等于计算出的状态，返回1.0
        3. 否则返回0.0
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            
        Returns:
            转移概率 (0.0 或 1.0)
        """
        # TODO: 在这里实现状态转移概率
        if action == (0, -1):
            next_state_calculated = (state[0]-1,state[1])
        elif action == (0, 1):
            next_state_calculated = (state[0]+1,state[1])
        elif action == (-1, 0):
            next_state_calculated = (state[0], state[1]-1)
        elif action == (1, 0):
            next_state_calculated = (state[0], state[1]+1)
        
        if next_state == next_state_calculated:
            return 1.0
        else:
            return 0.0
    
    def get_next_state(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        """
        获取执行动作后的下一个状态（确定性环境）
        
        TODO: 实现状态转移函数
        提示：
        1. 计算新状态 = state + action
        2. 如果新状态无效（出界或障碍物），返回原状态
        3. 否则返回新状态
        
        Args:
            state: 当前状态
            action: 执行的动作
            
        Returns:
            下一个状态
        """
        # TODO: 在这里实现状态转移函数
        if action == (0, -1):
            next_state_calculated = (state[0]-1,state[1])
        elif action == (0, 1):
            next_state_calculated = (state[0]+1,state[1])
        elif action == (-1, 0):
            next_state_calculated = (state[0], state[1]-1)
        elif action == (1, 0):
            next_state_calculated = (state[0], state[1]+1)

        if next_state_calculated in self.obstacles or next_state_calculated[0]<0 or next_state_calculated[0] >= self.grid_size[0] or next_state_calculated[1] <0 or next_state_calculated[1] >= self.grid_size[1]:
            return state
        else:
            return next_state_calculated


# ============================================================================
# 第二部分：定义策略（第1章内容）
# ============================================================================

def create_uniform_policy(mdp: GridWorldMDP) -> np.ndarray:
    """
    创建一个均匀随机策略 π(a|s)
    
    TODO: 实现均匀随机策略
    提示：
    - 返回一个形状为 (num_states, num_actions) 的数组
    - 对于每个状态，所有动作的概率相等（1/num_actions）
    
    Args:
        mdp: GridWorldMDP实例
        
    Returns:
        策略矩阵，shape=(num_states, num_actions)
    """
    # TODO: 在这里实现均匀随机策略
    num_states = mdp.grid_size[0] * mdp.grid_size[1]
    num_actions = len(mdp.actions)

    matrix = np.full((num_states, num_actions), 1.0 / num_actions)
    
    return matrix


# ============================================================================
# 第三部分：计算状态价值函数（第2章核心内容）
# ============================================================================

def compute_state_value_iterative(mdp: GridWorldMDP, policy: np.ndarray, 
                                   max_iterations: int = 1000, 
                                   tolerance: float = 1e-6) -> np.ndarray:
    """
    使用迭代法计算状态价值函数 v_π(s)
    
    这是贝尔曼方程的迭代求解方法：
    v_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ * v_π(s')]
    
    TODO: 实现迭代法计算状态价值函数
    提示：
    1. 初始化所有状态价值为0
    2. 迭代更新每个状态的价值：
       - 对于每个状态s，计算所有可能动作a的期望
       - 对于每个动作a，计算所有可能下一状态s'的期望
       - 使用贝尔曼方程更新v(s)
    3. 当价值函数变化小于tolerance时停止
    
    Args:
        mdp: GridWorldMDP实例
        policy: 策略矩阵，shape=(num_states, num_actions)
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        
    Returns:
        状态价值函数，shape=(num_states,)
    """
    # TODO: 在这里实现迭代法计算状态价值函数
    v = np.zeros(mdp.num_states)
    for iter in range(max_iterations):
        v_new = np.zeros(mdp.num_states)

        for idx_state in range(mdp.num_states):
            v_new_s = 0

            for idx_action, action in enumerate(mdp.actions):

                state = mdp.index_to_state(idx_state)
                next_state = mdp.get_next_state(state, action)

                reward = mdp.get_reward(state, action, next_state)

                p = mdp.get_transition_prob(state, action, next_state)

                v_new_s += policy[idx_state, idx_action] * p * (reward + mdp.gamma * v[mdp.state_to_index(next_state)])
            
            v_new[idx_state] = v_new_s
        
        gap = np.max(np.abs(v_new-v))

        v = v_new.copy()
        if gap < tolerance:
            break
    
    return v

def compute_state_value_matrix_form(mdp: GridWorldMDP, policy: np.ndarray) -> np.ndarray:
    """
    使用矩阵形式直接求解状态价值函数 v_π = (I - γP_π)^(-1) * r_π
    
    这是贝尔曼方程的矩阵形式解
    
    TODO: 实现矩阵形式求解状态价值函数
    提示：
    1. 构建状态转移概率矩阵 P_π (考虑策略)
    2. 构建奖励向量 r_π
    3. 使用公式 v_π = (I - γP_π)^(-1) * r_π 求解
    
    Args:
        mdp: GridWorldMDP实例
        policy: 策略矩阵，shape=(num_states, num_actions)
        
    Returns:
        状态价值函数，shape=(num_states,)
    """
    # TODO: 在这里实现矩阵形式求解状态价值函数
    P_pi = np.zeros((mdp.num_states, mdp.num_states))

    for state_idx in range(mdp.num_states):
        state = mdp.index_to_state(state_idx)

        for action_idx in range(len(mdp.actions)):
            action_prob = policy[state_idx][action_idx]
            action = mdp.actions[action_idx]

            next_state = mdp.get_next_state(state, action)
            next_state_idx = mdp.state_to_index(next_state)
            trans_prob = mdp.get_transition_prob(state, action, next_state)
            P_pi[state_idx, next_state_idx] += trans_prob * action_prob

    r_pi = np.zeros(mdp.num_states)

    for state_idx in range(mdp.num_states):
        state = mdp.index_to_state(state_idx)

        for action_idx in range(len(mdp.actions)):
            action_prob = policy[state_idx][action_idx]
            action = mdp.actions[action_idx]

            next_state = mdp.get_next_state(state, action)

            reward = mdp.get_reward(state, action, next_state)
            r_pi[state_idx] += reward * action_prob
    
    I_pi = np.eye(mdp.num_states)
    v_pi = np.linalg.solve(I_pi - mdp.gamma * P_pi, r_pi)

    return v_pi


# ============================================================================
# 第四部分：验证贝尔曼方程（第2章核心验证）
# ============================================================================

def verify_bellman_equation(mdp: GridWorldMDP, 
                           state_values: np.ndarray, 
                           policy: np.ndarray,
                           tolerance: float = 1e-4) -> Tuple[bool, float]:
    """
    验证计算出的状态价值函数是否满足贝尔曼方程
    
    对于每个状态s，检查：
    v_π(s) ≈ Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ * v_π(s')]
    
    TODO: 实现贝尔曼方程验证
    提示：
    1. 对于每个状态s，计算贝尔曼方程右侧的值
    2. 与左侧的v_π(s)比较
    3. 计算最大误差
    4. 如果最大误差小于tolerance，返回True，否则返回False
    
    Args:
        mdp: GridWorldMDP实例
        state_values: 状态价值函数，shape=(num_states,)
        policy: 策略矩阵，shape=(num_states, num_actions)
        tolerance: 允许的误差阈值
        
    Returns:
        (是否满足贝尔曼方程, 最大误差)
    """
    # TODO: 在这里实现贝尔曼方程验证
    v_pi_right_value = np.zeros(mdp.num_states)

    for state_idx in range(mdp.num_states):

        for action_idx, action in enumerate(mdp.actions):
            state = mdp.index_to_state(state_idx)
            next_state = mdp.get_next_state(state, action)
            next_state_idx = mdp.state_to_index(next_state)

            reward = mdp.get_reward(state, action, next_state)
            v_pi_next_state = state_values[next_state_idx]

            action_prob = policy[state_idx, action_idx]

            trans_prob = mdp.get_transition_prob(state, action, next_state)
            v_pi_right_value[state_idx] += action_prob * trans_prob* (reward + mdp.gamma * v_pi_next_state)

    max_error = np.max(np.abs(v_pi_right_value - state_values))
    if max_error < tolerance:
        return True, max_error
    else:
        return False, max_error


# ============================================================================
# 第五部分：可视化与测试
# ============================================================================

def visualize_state_values(mdp: GridWorldMDP, state_values: np.ndarray, 
                           title: str = "State Values"):
    """
    可视化状态价值函数
    
    Args:
        mdp: GridWorldMDP实例
        state_values: 状态价值函数
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 创建网格
    width, height = mdp.grid_size
    values_2d = np.zeros(mdp.grid_size)
    
    for i in range(mdp.num_states):
        state = mdp.index_to_state(i)
        values_2d[state] = state_values[i]
    
    # 绘制热力图
    im = ax.imshow(values_2d.T, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax)
    
    # 添加数值标注
    for i in range(mdp.num_states):
        state = mdp.index_to_state(i)
        x, y = state
        ax.text(x, y, f'{state_values[i]:.2f}', 
                ha='center', va='center', color='white', fontsize=10)
    
    # 标记起始、目标和障碍物
    sx, sy = mdp.start_state
    ax.plot(sx, sy, 'go', markersize=15, label='Start')
    
    tx, ty = mdp.target_state
    ax.plot(tx, ty, 'r*', markersize=20, label='Target')
    
    for obs in mdp.obstacles:
        ox, oy = obs
        ax.plot(ox, oy, 'ks', markersize=15, label='Obstacle')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：运行完整的验证流程
    """
    print("=" * 60)
    print("编程作业：MDP与贝尔曼方程验证")
    print("=" * 60)
    
    # 1. 创建MDP环境
    print("\n[步骤1] 创建Grid World MDP环境...")
    mdp = GridWorldMDP()
    print(f"  网格大小: {mdp.grid_size}")
    print(f"  状态数量: {mdp.num_states}")
    print(f"  动作数量: {len(mdp.actions)}")
    print(f"  折扣因子: {mdp.gamma}")
    
    # 2. 创建策略
    print("\n[步骤2] 创建均匀随机策略...")
    policy = create_uniform_policy(mdp)
    print(f"  策略形状: {policy.shape}")
    print(f"  策略示例（状态0）: {policy[0]}")
    
    # 3. 使用迭代法计算状态价值
    print("\n[步骤3] 使用迭代法计算状态价值函数...")
    state_values_iter = compute_state_value_iterative(mdp, policy)
    print(f"  迭代法计算完成")
    print(f"  状态价值示例（前5个状态）: {state_values_iter[:5]}")
    
    # 4. 使用矩阵形式计算状态价值（可选，用于验证）
    print("\n[步骤4] 使用矩阵形式计算状态价值函数（验证用）...")
    try:
        state_values_matrix = compute_state_value_matrix_form(mdp, policy)
        print(f"  矩阵法计算完成")
        print(f"  两种方法的最大差异: {np.max(np.abs(state_values_iter - state_values_matrix))}")
    except Exception as e:
        print(f"  矩阵法计算失败（可能因为矩阵不可逆）: {e}")
        state_values_matrix = state_values_iter
    
    # 5. 验证贝尔曼方程
    print("\n[步骤5] 验证贝尔曼方程...")
    is_satisfied, max_error = verify_bellman_equation(mdp, state_values_iter, policy)
    print(f"  贝尔曼方程是否满足: {is_satisfied}")
    print(f"  最大误差: {max_error:.6f}")
    
    # 6. 可视化结果
    print("\n[步骤6] 可视化状态价值函数...")
    visualize_state_values(mdp, state_values_iter, "State Values (Iterative Method)")
    
    # 7. 输出成功标准
    print("\n" + "=" * 60)
    print("验证结果总结:")
    print("=" * 60)
    if is_satisfied and max_error < 1e-4:
        print("✓ 成功！状态价值函数满足贝尔曼方程（误差 < 1e-4）")
        print("✓ 作业完成标准：已达成")
    else:
        print("✗ 失败！状态价值函数不满足贝尔曼方程")
        print(f"  当前误差: {max_error:.6f}")
        print("  请检查实现是否正确")
    print("=" * 60)


if __name__ == "__main__":
    main()

