import numpy as np
import os
from gridworld import GridWorld
from sarsa import sarsa
from qlearning import q_learning
import sys
import io

# Memastikan stdout mendukung utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# Membuat folder 'result' jika belum ada
if not os.path.exists('result'):
    os.makedirs('result')

# Map konfigurasi 4x4
map_config = {
    "width": 4,  # Lebar grid
    "height": 4,  # Tinggi grid
    "start": (0, 0),  # Posisi start
    "goal": (3, 3),  # Posisi goal
    "obstacles": [(1, 1), (1, 2), (2, 1)]  # Posisi obstacles
}

# Hyperparameter untuk RL
episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Fungsi untuk menampilkan peta arah hasil policy
def plot_direction_map(Q, env):
    direction_map = np.full((env.height, env.width), ' ', dtype='<U2')

    for x in range(env.height):
        for y in range(env.width):
            if (x, y) == env.goal:
                direction_map[x, y] = 'G'  # Tanda Goal
            elif (x, y) in env.obstacles:
                direction_map[x, y] = 'X'  # Tanda Obstacle
            else:
                best_action = np.argmax(Q[x, y])
                direction_map[x, y] = ['↑', '↓', '←', '→'][best_action]  # Arah yang terbaik

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