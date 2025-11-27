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
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        
        self.memory = deque(maxlen=1000)  # Smaller memory for faster learning
        self.batch_size = 16  # Smaller batch for faster updates
        
        self.episode_count = 0
        self.best_score = 0
        
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
        """Update weights using experience replay - acts as Lamarckian weight adjustment"""
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
            
            # Update Q value for the taken action
            current_q[action] = current_q[action] + self.learning_rate * (target - current_q[action])
            
            # Update network weights using gradient approximation
            self.update_network(state, current_q)
    
    def update_network(self, state, target_output):
        """Update neural network weights using simple gradient approximation"""
        current_output = self.genome.feed_forward(state)
        
        # Convert to numpy arrays for subtraction
        target_output = np.array(target_output)
        current_output = np.array(current_output)
        error = target_output - current_output
        
        # Update weights in all connections
        for conn in self.genome.connections.values():
            if conn.enabled:
                # Simple gradient approximation
                input_node = self.genome.nodes[conn.in_node]
                # Use the error for the action with highest target
                gradient = input_node.value * error[np.argmax(target_output)] * self.learning_rate * 0.1
                conn.weight += gradient
                # Keep weights in reasonable range
                conn.weight = max(-2, min(2, conn.weight))
    
    def calculate_reward(self, game, score, game_over, steps_since_food):
        """Calculate reward for current state"""
        reward = 0
        
        if game_over:
            reward = -50  # Penalty for dying
        elif score > self.best_score:
            reward = 50   # Reward for new high score
            self.best_score = score
        else:
            # Small rewards for staying alive and finding food
            reward = 0.1
            
            # Bonus for efficient movement
            if steps_since_food < 10:
                reward += 1
                
            # Penalty for wandering too long without food
            if steps_since_food > 30:
                reward -= 0.5
        
        return reward
    
    def train_single_episode(self, render=False, speed=1000):
        """Train for one episode and return the final score"""
        from game import SnakeGame
        
        game = SnakeGame(w=500, h=500)
        
        max_steps_without_food = 50
        steps_since_food = 0
        total_steps = 0
        initial_score = 0
        
        state = self.snake_ai.get_state(game)
        
        while True:
            action = self.choose_action(state, training=True)
            current_direction = game.direction
            new_direction = self.snake_ai.action_to_direction(action, current_direction)
            game.pressed_direction = new_direction
            
            game_over, score = game.play_step()
            total_steps += 1
            
            if score > initial_score:
                steps_since_food = 0
                initial_score = score
                max_steps_without_food = min(50 + score, 361)
            else:
                steps_since_food += 1
            
            reward = self.calculate_reward(game, score, game_over, steps_since_food)
            
            next_state = self.snake_ai.get_state(game) if not game_over else None
            
            self.remember(state, action, reward, next_state, game_over)
            
            # Update weights after each step for faster learning
            if len(self.memory) >= self.batch_size:
                self.replay()
            
            state = next_state
            
            if game_over or steps_since_food > max_steps_without_food:
                break
        
        # Final weight update with all experiences
        if len(self.memory) >= self.batch_size:
            self.replay()
        
        self.exploration_rate = max(self.min_exploration, 
                                  self.exploration_rate * self.exploration_decay)
        
        self.episode_count += 1
        
        return score
    
    def fine_tune_genome(self, episodes=3):
        """Fine-tune a genome for a few episodes - used during NEAT fitness evaluation"""
        original_exploration = self.exploration_rate
        self.exploration_rate = 0.1  # Low exploration for fine-tuning
        
        best_score = 0
        for _ in range(episodes):
            score = self.train_single_episode(render=False, speed=1000)
            if score > best_score:
                best_score = score
        
        # Restore original exploration rate
        self.exploration_rate = original_exploration
        
        return best_score
    
    def train(self, episodes=100, render=True, speed=50, save_interval=10):
        """Train for multiple episodes"""
        print(f"Starting RL training: {episodes} episodes")
        print(f"Learning rate: {self.learning_rate}, Discount: {self.discount_factor}")
        
        start_time = time.time()
        best_score = 0
        
        for episode in range(episodes):
            score = self.train_single_episode(render=render, speed=speed)
            
            if score > best_score:
                best_score = score
                self.save(f"best_rl_snake_ep{episode}.pkl")
            
            if (episode + 1) % 10 == 0 or episode == 0:
                print(f"Episode {episode + 1:4d}: Score={score:3d}, "
                      f"Explore={self.exploration_rate:.3f}")
            
            if (episode + 1) % save_interval == 0:
                self.save(f"rl_snake_checkpoint_ep{episode + 1}.pkl")
        
        self.save("rl_snake_final.pkl")
        
        elapsed_time = time.time() - start_time
        print(f"RL training completed")
        print(f"Episodes: {episodes}, Best score: {best_score}")
        print(f"Final explore: {self.exploration_rate:.4f}, Time: {elapsed_time:.2f}s")
        
        return best_score
    
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