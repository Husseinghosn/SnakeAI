import numpy as np
from collections import namedtuple

Point = namedtuple('Point', 'x, y')

class GridProcessor:
    def __init__(self, grid_size=19, block_size=20):
        self.grid_size = grid_size
        self.block_size = block_size

    def create_merged_grid(self, snake, food):
        """Create a single 19x19 grid with snake body and food"""
        grid = np.full((self.grid_size, self.grid_size), 0)  # 0 for empty space
        
        # Add snake body (head = 1, body segments increase by 1)
        for i, segment in enumerate(snake):
            grid_x = int(segment.x // self.block_size)
            grid_y = int(segment.y // self.block_size)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_y][grid_x] = i + 1
        
        if food:
            food_x = int(food.x // self.block_size)
            food_y = int(food.y // self.block_size)
            if 0 <= food_x < self.grid_size and 0 <= food_y < self.grid_size:
                if grid[food_y][food_x] == 0:
                    grid[food_y][food_x] = -1
        
        return grid

    def rotate_grid(self, grid, direction):
        if direction.name == 'UP':
            return grid  # No rotation
        elif direction.name == 'RIGHT':
            return np.rot90(grid, k=1)  # Rotate 90° clockwise
        elif direction.name == 'DOWN':
            return np.rot90(grid, k=2)  # Rotate 180°
        elif direction.name == 'LEFT':
            return np.rot90(grid, k=3)  # Rotate 90° counter-clockwise
        return grid

    def center_grid(self, grid):
        """Center the grid on the snake's head"""
        # Find head position (value = 1)
        head_positions = np.where(grid == 1)
        if len(head_positions[0]) == 0:
            return grid  # No head found
            
        head_y, head_x = head_positions[0][0], head_positions[1][0]
        center = self.grid_size // 2
        
        # Calculate shift needed to center head
        shift_y = center - head_y
        shift_x = center - head_x
        
        # Shift grid
        centered = np.roll(grid, shift_y, axis=0)
        centered = np.roll(centered, shift_x, axis=1)
        
        return centered

    def get_normalized_input(self, snake, food, direction):
        """Create normalized input grid for neural network"""
        # Create single merged grid
        merged_grid = self.create_merged_grid(snake, food)
        
        # Rotate based on snake's direction
        rotated_grid = self.rotate_grid(merged_grid, direction)
        
        # Center on snake head
        centered_grid = self.center_grid(rotated_grid)
        
        return {"grid": centered_grid}