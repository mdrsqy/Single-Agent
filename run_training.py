import numpy as np
import os
from gridworld import GridWorld
from sarsa import sarsa
from qlearning import q_learning
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

if not os.path.exists('result'):
    os.makedirs('result')

map_config = {
    "width": 4,
    "height": 4,
    "start": (0, 0),
    "goal": (3, 3),
    "obstacles": [(1, 1), (2, 2)]
}

episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

def plot_direction_map(Q, env):
    direction_map = np.full((env.height, env.width), ' ', dtype='<U2')

    for x in range(env.height):
        for y in range(env.width):
            if (x, y) == env.goal:
                direction_map[x, y] = '⛳'
            elif (x, y) in env.obstacles:
                direction_map[x, y] = '💣'
            else:
                best_action = np.argmax(Q[x, y])
                direction_map[x, y] = ['⬆️', '⬇️', '⬅️', '➡️'][best_action]

    for row in direction_map:
        print(" ".join(row))

# Menjalankan eksperimen pada Map 1
env = GridWorld(**map_config)

print(f"\n=== MAP 1 ===")

# Mulai SARSA Training
print("Starting SARSA Training...")
Q_sarsa, time_sarsa, action_sarsa = sarsa(env, episodes, alpha, gamma, epsilon, save_path="result/sarsa_map1.npy")
print(f"SARSA Q-values saved for Map 1")
print(f"Training Time (SARSA): {time_sarsa:.2f} seconds")
print(f"Action Counts (SARSA): {action_sarsa}")

# Mulai Q-Learning Training
print("Starting Q-Learning Training...")
Q_qlearn, time_qlearn, action_qlearn = q_learning(env, episodes, alpha, gamma, epsilon, save_path="result/qlearning_map1.npy")
print(f"Q-Learning Q-values saved for Map 1")
print(f"Training Time (Q-Learning): {time_qlearn:.2f} seconds")
print(f"Action Counts (Q-Learning): {action_qlearn}")

# Menampilkan peta arah SARSA
print("Direction Map (SARSA):")
plot_direction_map(Q_sarsa, env)

# Menampilkan peta arah Q-Learning
print("\nDirection Map (Q-Learning):")
plot_direction_map(Q_qlearn, env)

# Jika ingin load kembali:
# Q_sarsa_loaded = np.load("result/sarsa_map1.npy")
# Q_qlearn_loaded = np.load("result/qlearning_map1.npy")