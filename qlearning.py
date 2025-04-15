import numpy as np
import random
import time

def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Random action
    return np.argmax(Q[state[0], state[1]])  # Greedy action

def q_learning(env, episodes, alpha, gamma, epsilon, save_path=None):
    Q = np.zeros((env.height, env.width, 4))
    action_counts = np.zeros(4)  # To track action choices
    
    start_time = time.time()

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(action)

            best_next_action = np.argmax(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], best_next_action]
                - Q[state[0], state[1], action]
            )

            action_counts[action] += 1  # Track action

            state = next_state

    training_time = time.time() - start_time
    if save_path:
        np.save(save_path, Q)  # Save Q-table

    return Q, training_time, action_counts