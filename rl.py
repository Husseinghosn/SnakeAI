# rl.py
import pygame
import numpy as np
import random
import time
import copy
from collections import deque
from snake_ai import SnakeAI

class ReinforcementTrainer:
    def __init__(self, genome, learning_rate=0.01, discount_factor=0.95, exploration_rate=0.1):
        self.genome = genome
        self.original_genome = copy.deepcopy(genome)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        self.episode_count = 0
        self.total_reward = 0
        self.best_score = 0
        self.reward_history = []
        self.score_history = []
        
        self.snake_ai = SnakeAI()
        
    def choose_action(self, state, training=True):
        if training and random.random() < self.exploration_rate:
            return random.randint(0, 2)
        else:
            output = self.genome.feed_forward(state)
            return np.argmax(output)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            current_q = self.genome.feed_forward(state)
            
            if done:
                target = reward
            else:
                next_q = self.genome.feed_forward(next_state)
                max_next_q = np.max(next_q)
                target = reward + self.discount_factor * max_next_q
            
            current_q[action] = current_q[action] + self.learning_rate * (target - current_q[action])
            
            self.update_network(state, current_q)
    
    def update_network(self, state, target_output):
        current_output = self.genome.feed_forward(state)
        error = target_output - current_output
        
        for conn in self.genome.connections.values():
            if conn.enabled:
                input_node = self.genome.nodes[conn.in_node]
                gradient = input_node.value * error[np.argmax(target_output)] * 0.001
                conn.weight += gradient
                conn.weight = max(-2, min(2, conn.weight))
    
    def calculate_reward(self, game, score, game_over, steps_since_food):
        reward = 0
        
        if game_over:
            reward = -100
        elif score > self.best_score:
            reward = 100
            self.best_score = score
        else:
            reward = 1
            
            if steps_since_food < 10:
                reward += 2
                
            if steps_since_food > 50:
                reward -= 1
        
        return reward
    
    def train_episode(self, render=True, speed=50):
        from game import SnakeGame
        
        game = SnakeGame(w=500, h=500)
        
        max_steps_without_food = 50
        steps_since_food = 0
        total_steps = 0
        initial_score = 0
        episode_reward = 0
        
        state = self.snake_ai.get_state(game)
        
        while True:
            if render:
                pygame.time.delay(1000 // speed)
            
            action = self.choose_action(state, training=True)
            current_direction = game.direction
            new_direction = self.snake_ai.action_to_direction(action, current_direction)
            game.pressed_direction = new_direction
            
            old_score = game.score
            game_over, score = game.play_step()
            total_steps += 1
            
            if score > initial_score:
                steps_since_food = 0
                initial_score = score
                max_steps_without_food = min(50 + score, 625)
            else:
                steps_since_food += 1
            
            reward = self.calculate_reward(game, score, game_over, steps_since_food)
            episode_reward += reward
            
            next_state = self.snake_ai.get_state(game) if not game_over else None
            
            self.remember(state, action, reward, next_state, game_over)
            
            state = next_state
            
            if game_over or steps_since_food > max_steps_without_food:
                break
        
        self.replay()
        
        self.exploration_rate = max(self.min_exploration, 
                                  self.exploration_rate * self.exploration_decay)
        
        self.episode_count += 1
        self.total_reward += episode_reward
        self.reward_history.append(episode_reward)
        self.score_history.append(score)
        
        return score, episode_reward, total_steps
    
    def train(self, episodes=100, render=True, speed=50, save_interval=10):
        print(f"Starting RL training: {episodes} episodes")
        print(f"Learning rate: {self.learning_rate}, Discount: {self.discount_factor}")
        
        start_time = time.time()
        best_score = 0
        
        for episode in range(episodes):
            score, reward, steps = self.train_episode(render=render, speed=speed)
            
            if score > best_score:
                best_score = score
                self.save(f"best_rl_snake_ep{episode}.pkl")
            
            if (episode + 1) % 10 == 0 or episode == 0:
                avg_reward = np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else reward
                avg_score = np.mean(self.score_history[-10:]) if len(self.score_history) >= 10 else score
                print(f"Episode {episode + 1:4d}: Score={score:3d}, "
                      f"Reward={reward:6.1f}, "
                      f"Explore={self.exploration_rate:.3f}")
            
            if (episode + 1) % save_interval == 0:
                self.save(f"rl_snake_checkpoint_ep{episode + 1}.pkl")
        
        self.save("rl_snake_final.pkl")
        
        elapsed_time = time.time() - start_time
        print(f"RL training completed")
        print(f"Episodes: {episodes}, Best score: {best_score}")
        print(f"Final explore: {self.exploration_rate:.4f}, Time: {elapsed_time:.2f}s")
        
        return best_score
    
    def evaluate(self, num_episodes=10, render=True, speed=50):
        print(f"Evaluating: {num_episodes} episodes")
        
        scores = []
        total_rewards = []
        
        for episode in range(num_episodes):
            original_exploration = self.exploration_rate
            self.exploration_rate = 0
            
            score, reward, steps = self.train_episode(render=render, speed=speed, training=False)
            
            self.exploration_rate = original_exploration
            
            scores.append(score)
            total_rewards.append(reward)
            
            print(f"Eval {episode + 1}: Score={score}, Reward={reward:.1f}")
        
        avg_score = np.mean(scores)
        avg_reward = np.mean(total_rewards)
        
        print(f"Eval Results - Avg Score: {avg_score:.2f}, Best: {max(scores)}")
        
        return scores, total_rewards
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            import pickle
            pickle.dump(self.genome, f)
        print(f"Saved: {filename}")
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            import pickle
            self.genome = pickle.load(f)
        print(f"Loaded: {filename}")