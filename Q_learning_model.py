import numpy as np

class QLearningModel:
    def __init__(self, state_space, action_space, alpha=0.5, gamma=0.9):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((state_space, action_space))
        self.alpha = alpha
        self.gamma = gamma

    def update_q_table(self, current_state, action, reward, next_state):
        current_q_value = self.q_table[current_state][action]
        max_future_q_value = np.max(self.q_table[next_state])
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_future_q_value)
        self.q_table[current_state][action] = new_q_value
