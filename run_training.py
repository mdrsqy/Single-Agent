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

maps = {
    "Map 1": {
        "width": 4,
        "height": 4,
        "start": (0, 0),
        "goal": (3, 3),
        "obstacles": [(1, 1), (2, 2)]
    },
    "Map 2": {
        "width": 5,
        "height": 5,
        "start": (0, 0),
        "goal": (4, 4),
        "obstacles": [(0, 4), (1, 1), (1, 4), (2, 2), (3, 0), (3, 4), (4, 2)]
    },
    "Map 3": {
        "width": 8,
        "height": 8,
        "start": (0, 0),
        "goal": (7, 7),
        "obstacles": [(2, 3), (3, 5), (4, 3), (5, 1), (5, 2), (5, 6), (6, 1), (6, 4), (6, 6), (7, 3)]
    }
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
for map_name, map_config in maps.items():
    env = GridWorld(**map_config)
    
    print(f"\n=== {map_name} ===")

    # Mulai SARSA Training
    print("Starting SARSA Training...")
    Q_sarsa, time_sarsa, action_sarsa = sarsa(env, episodes, alpha, gamma, epsilon, save_path=f"result/sarsa_{map_name}.npy")
    print(f"SARSA Q-values saved for {map_name}")
    print(f"Training Time (SARSA): {time_sarsa:.2f} seconds")
    print(f"Action Counts (SARSA): {action_sarsa}")

    # Mulai Q-Learning Training
    print("Starting Q-Learning Training...")
    Q_qlearn, time_qlearn, action_qlearn = q_learning(env, episodes, alpha, gamma, epsilon, save_path=f"result/qlearning_{map_name}.npy")
    print(f"Q-Learning Q-values saved for {map_name}")
    print(f"Training Time (Q-Learning): {time_qlearn:.2f} seconds")
    print(f"Action Counts (Q-Learning): {action_qlearn}")

    # Menampilkan peta arah SARSA
    print("Direction Map (SARSA):")
    plot_direction_map(Q_sarsa, env)

    # Menampilkan peta arah Q-Learning
    print("\nDirection Map (Q-Learning):")
    plot_direction_map(Q_qlearn, env)

    # Jika ingin load kembali:
    # Q_sarsa_loaded = np.load(f"result/sarsa_{map_name}.npy")
    # Q_qlearn_loaded = np.load(f"result/qlearning_{map_name}.npy")