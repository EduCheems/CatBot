import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

def manhattan_distance(state: int) -> int:
    # br bc = bot row bot column
    # cr cc = cat row cat column
    # |bot row - cat row| + |bot col - cat col|
    
    # Extract positions from the state integer (Format: RRCCrrcc)
    br = (state // 1000) % 10
    bc = (state // 100) % 10
    cr = (state // 10) % 10
    cc = state % 10
    
    # Fixed typo from original snippet: changed ar/ac to br/bc
    return abs(br - cr) + abs(bc - cc)

#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project 
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################

    # Learning Rate (Alpha): Controls how much new info overrides old info.
    learning_rate = 0.2 
    
    # Discount Factor (Gamma): Importance of future rewards (high for long-term goal).
    discount_factor = 0.99 
    
    # Exploration Rate (Epsilon): Probability of choosing a random action.
    exploration_rate = 1.0 
    
    # Minimum Exploration: Keeps a small chance of exploration to prevent getting stuck.
    explore_rate_minimum = 0.01 
    
    # Decay Rate: How fast we reduce exploration. 
    # 0.999 * 5000 episodes reduces epsilon nicely to near minimum.
    exploration_decay = 0.999 

    # Limit steps per episode during training to prevent infinite loops early on
    max_steps_per_ep = 60 # Matches the evaluation limit 

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 

        # 1. Reset the environment
        curr_state, _ = env.reset()
        prev_distance = manhattan_distance(curr_state)
        done = False
        steps = 0

        while not done and steps < max_steps_per_ep:
            # 2. Decide whether to explore (random) or exploit (best known action)
            if random.random() < exploration_rate:
                action = env.action_space.sample() # Random action (0-4) [cite: 86]
            else:
                # Choose the action with the highest Q-value for the current state
                action = np.argmax(q_table[curr_state])

            # 3. Take action
            next_state, _, done, _, _ = env.step(action)
            steps += 1

            # 4. Compute Reward Manually [cite: 88, 98]
            # The environment returns 0 by default, so we engineer a reward structure.
            reward = 0
            curr_distance = manhattan_distance(next_state)

            if done:
                # Big reward for catching the cat
                reward = 100 
            else:
                # Small penalty for every time step to encourage speed
                reward -= 0.1
                
                # Reward Shaping: Guide the bot towards the cat
                if curr_distance < prev_distance:
                    reward += 1.0  # Reward moving closer
                elif curr_distance > prev_distance:
                    reward -= 1.5  # Penalize moving further away
                else:
                    reward -= 0.5  # Penalize staying at the same distance (e.g., waiting)

            # 5. Update Q-table (Bellman Equation)
            # Q(s,a) = (1-lr) * Q(s,a) + lr * (reward + gamma * max(Q(s',a')))
            
            old_value = q_table[curr_state][action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - learning_rate) * old_value + \
                        learning_rate * (reward + discount_factor * next_max)
            
            q_table[curr_state][action] = new_value

            # Update state and distance for next iteration
            curr_state = next_state
            prev_distance = curr_distance

        # Decay exploration rate after each episode
        if exploration_rate > explore_rate_minimum:
            exploration_rate *= exploration_decay
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table