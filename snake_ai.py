# snake_ai.py
import pygame
import numpy as np
from game import SnakeGame, Direction

class SnakeAI:
    def __init__(self):
        self.input_size = 1250
        self.output_size = 3
        
    def get_state(self, game):
        """Get game state as neural network input"""
        ai_input = game.get_ai_input()
        snake_grid_flat = ai_input["snake_grid"].flatten()
        food_grid_flat = ai_input["food_grid"].flatten()
        network_input = np.concatenate([snake_grid_flat, food_grid_flat])
        network_input = np.clip(network_input, -1, 10) / 10.0
        return network_input
    
    def action_to_direction(self, action, current_direction):
        """Convert action index to direction"""
        if action == 0:
            return current_direction
        elif action == 1:
            if current_direction == Direction.RIGHT:
                return Direction.DOWN
            elif current_direction == Direction.DOWN:
                return Direction.LEFT
            elif current_direction == Direction.LEFT:
                return Direction.UP
            else:
                return Direction.RIGHT
        else:
            if current_direction == Direction.RIGHT:
                return Direction.UP
            elif current_direction == Direction.UP:
                return Direction.LEFT
            elif current_direction == Direction.LEFT:
                return Direction.DOWN
            else:
                return Direction.RIGHT
    
    def play_game(self, genome, render=True, speed=50, training=False):
        """Play a single game with given genome"""
        game = SnakeGame(w=380, h=380)
        
        max_steps_without_food = 50
        steps_since_food = 0
        total_steps = 0
        initial_score = 0

        last_turns = []
        four_left_turns = 0
        four_right_turns = 0

        # Track consecutive steps in the same direction
        same_dir_steps = 0
        last_direction = None
        over_25_same_dir_count = 0
        
        while True:
            if render:
                pygame.time.delay(1000 // speed)
            
            state = self.get_state(game)
            output = genome.feed_forward(state)
            action = np.argmax(output)
            
            current_direction = game.direction
            new_direction = self.action_to_direction(action, current_direction)
            game.pressed_direction = new_direction

            # Track turn direction
            turn = 0
            if action == 1:
                turn = 1  # left
            elif action == 2:
                turn = -1  # right
            last_turns.append(turn)
            if len(last_turns) > 4:
                last_turns.pop(0)
            if last_turns == [1, 1, 1, 1]:
                four_left_turns += 1
            if last_turns == [-1, -1, -1, -1]:
                four_right_turns += 1

            # Track consecutive same direction steps
            if last_direction is None or new_direction == last_direction:
                same_dir_steps += 1
                if same_dir_steps == 26:  # Only count once per streak
                    over_25_same_dir_count += 1
            else:
                same_dir_steps = 1
            last_direction = new_direction
            
            game_over, score = game.play_step()
            total_steps += 1
            
            if score > initial_score:
                steps_since_food = 0
                initial_score = score
                max_steps_without_food = min(50 + score, 361)
            else:
                steps_since_food += 1
            
            if game_over or steps_since_food > max_steps_without_food:
                break
        
        return score, total_steps, four_left_turns, four_right_turns, over_25_same_dir_count

    def calculate_fitness(self, score, total_steps, steps_since_food, max_steps_without_food, four_left_turns=0, four_right_turns=0, over_25_same_dir_count=0):
        """Calculate fitness score"""
        fitness = score * 1000
        
        if score > 0:
            efficiency = total_steps / score
            efficiency_bonus = max(0, 500 - efficiency)
            fitness += efficiency_bonus
        
        survival_bonus = total_steps * 0.1 * (1 + score * 0.1)
        fitness += survival_bonus
        
        if score >= 5:
            fitness += 2000
        if score >= 10:
            fitness += 5000
        if score >= 20:
            fitness += 10000
        if score >= 30:
            fitness += 20000
        if score >= 40:
            fitness += 50000
        if score >= 50:
            fitness += 100000
            
        wandering_penalty = steps_since_food * 0.5
        fitness = max(fitness - wandering_penalty, 1.0)

        # Penalty for four consecutive left or right turns
        penalty = 500 * (four_left_turns + four_right_turns)
        fitness = max(fitness - penalty, 1.0)

        # Penalty for moving in the same direction for over 25 steps
        same_dir_penalty = 1000 * over_25_same_dir_count
        fitness = max(fitness - same_dir_penalty, 1.0)
        
        return fitness