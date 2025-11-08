import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class SnakeAI(nn.Module):
    def __init__(self, grid_size=25, hidden_size=256, hidden_layers=3):
        super(SnakeAI, self).__init__()
        self.grid_size = grid_size
        self.input_size = 2 * grid_size * grid_size  # snake grid + food grid
        
        # Build hidden layers dynamically
        layers = []
        prev_size = self.input_size
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, 3)  # 3 actions: forward, left, right
        
    def forward(self, snake_grid, food_grid):
        # Flatten both grids and concatenate
        snake_flat = snake_grid.view(snake_grid.size(0), -1)
        food_flat = food_grid.view(food_grid.size(0), -1)
        x = torch.cat([snake_flat, food_flat], dim=1)
        
        # Pass through network
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        return x
    
    def get_action(self, snake_grid, food_grid, epsilon=0.0):
        """Get action with optional exploration"""
        if random.random() < epsilon:
            return random.randint(0, 2)
        
        with torch.no_grad():
            output = self.forward(snake_grid, food_grid)
            return torch.argmax(output, dim=1).item()

class GeneticTrainer:
    def __init__(self, population_size=50, mutation_rate=0.2, grid_size=25):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.grid_size = grid_size
        
        self.population = []
        self.fitness_scores = []
        
        # Initialize population
        for _ in range(population_size):
            individual = SnakeAI(grid_size)
            self.population.append(individual)
    
    def evaluate_fitness(self, individual, game, num_games=3):
        """Evaluate individual by playing multiple games"""
        total_score = 0
        
        for _ in range(num_games):
            score = self._play_single_game(individual, game)
            total_score += score
        
        return total_score / num_games
    
    def _play_single_game(self, individual, original_game, max_steps=1000):
        """Play one game and return fitness score"""
        # Create a new game instance for evaluation
        from game import SnakeGame
        game = SnakeGame(w=500, h=500)
        
        # Copy initial state
        game.direction = original_game.direction
        game.snake = original_game.snake.copy()
        game.head = original_game.head
        game.food = original_game.food
        game.score = original_game.score
        
        steps = 0
        apples = 0
        steps_since_apple = 0
        steps_before_first_apple = 0
        got_first_apple = False
        
        while steps < max_steps:
            # Get AI input from current game state
            ai_input = game.get_ai_input()
            body_grid = torch.FloatTensor(ai_input['snake_grid']).unsqueeze(0)
            food_grid = torch.FloatTensor(ai_input['food_grid']).unsqueeze(0)
            
            # Get action from AI
            action = individual.get_action(body_grid, food_grid)
            
            # Map action to direction
            current_dir = game.direction
            if action == 0:  # Forward
                new_dir = current_dir
            elif action == 1:  # Turn right
                dir_map = {
                    game.direction.RIGHT: game.direction.DOWN,
                    game.direction.DOWN: game.direction.LEFT, 
                    game.direction.LEFT: game.direction.UP,
                    game.direction.UP: game.direction.RIGHT
                }
                new_dir = dir_map[current_dir]
            else:  # Turn left (action == 2)
                dir_map = {
                    game.direction.RIGHT: game.direction.UP,
                    game.direction.UP: game.direction.LEFT,
                    game.direction.LEFT: game.direction.DOWN,
                    game.direction.DOWN: game.direction.RIGHT
                }
                new_dir = dir_map[current_dir]
            
            # Update game direction and move
            game.direction = new_dir
            game._move(new_dir)
            game.snake.insert(0, game.head)
            
            # Check collision
            if game._is_collision():
                break
                
            # Check food
            if game.head == game.food:
                apples += 1
                game.score += 1
                game._place_food()
                steps_since_apple = 0
                
                # Track steps before first apple
                if not got_first_apple:
                    got_first_apple = True
                    steps_before_first_apple = steps
            else:
                game.snake.pop()
                steps_since_apple += 1
            
            steps += 1
            
            # Stop if stuck
            if steps_since_apple > 200:
                break
        
        # COMBINED FITNESS FUNCTION: Options 1, 2, and 3
        
        # Option 1: Base fitness with simple time penalty
        base_fitness = apples * 100 - steps * 0.5
        
        # Option 2: Progressive penalty (penalty increases over time)
        progressive_penalty = (steps * 0.02) ** 1.5
        
        # Option 3: Heavy penalty for not getting first apple quickly
        first_apple_penalty = 0
        if not got_first_apple:
            # Heavy penalty for not getting any apples
            first_apple_penalty = steps * 1
        else:
            # Moderate penalty for taking too long to get first apple
            if steps_before_first_apple > 50:  # If it takes more than 50 steps to get first apple
                first_apple_penalty = (steps_before_first_apple - 50) * 0.1
        
        # Efficiency bonus for high apple-to-step ratio
        efficiency_bonus = 0
        if steps > 0 and apples > 0:
            efficiency = apples / steps
            efficiency_bonus = efficiency * 50
        
        # Combine all components
        fitness = (base_fitness - progressive_penalty - first_apple_penalty + efficiency_bonus)
        
        # Debug output (uncomment to see fitness breakdown)
        # if apples > 0:
        #     print(f"Apples: {apples}, Steps: {steps}, Base: {base_fitness:.2f}, "
        #           f"ProgPen: {progressive_penalty:.2f}, FirstApplePen: {first_apple_penalty:.2f}, "
        #           f"EffBonus: {efficiency_bonus:.2f}, Total: {fitness:.2f}")
        
        return fitness
    
    def tournament_selection(self, tournament_size=3):
        """Select parents using tournament selection"""
        candidates = list(zip(self.population, self.fitness_scores))
        tournament = random.sample(candidates, tournament_size)
        return max(tournament, key=lambda x: x[1])[0]
    
    def crossover(self, parent1, parent2):
        """Create child by crossover of two parents"""
        child = SnakeAI(self.grid_size)
        child_sd = child.state_dict()
        p1_sd = parent1.state_dict()
        p2_sd = parent2.state_dict()
        
        for key in child_sd.keys():
            if 'weight' in key or 'bias' in key:
                # Uniform crossover
                mask = torch.rand_like(child_sd[key]) > 0.5
                child_sd[key] = torch.where(mask, p1_sd[key], p2_sd[key])
        
        child.load_state_dict(child_sd)
        return child
    
    def mutate(self, individual):
        """Apply mutation to weights and biases"""
        sd = individual.state_dict()
        
        for key in sd.keys():
            if 'weight' in key or 'bias' in key:
                mutation_mask = torch.rand_like(sd[key]) < self.mutation_rate
                mutation_strength = torch.randn_like(sd[key]) * 0.1
                sd[key] = sd[key] + mutation_mask * mutation_strength
        
        individual.load_state_dict(sd)
        return individual
    
    def create_new_generation(self):
        """Create new generation using elitism, crossover and mutation"""
        new_population = []
        
        # Elitism: keep top 20%
        elite_count = max(1, self.population_size // 5)
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Create offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.fitness_scores = []

def load_trained_model(model_path="best_snake_ai.pth"):
    """Load a trained model"""
    model = SnakeAI()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Loaded trained model!")
    else:
        print("No trained model found, using random weights.")
    return model