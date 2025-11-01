import numpy as np
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class GridProcessor:
    def __init__(self, grid_size=25, block_size=20):
        self.grid_size = grid_size
        self.block_size = block_size

    def create_snake_grid(self, snake):
        grid = np.full((self.grid_size, self.grid_size), -1)
        for i, segment in enumerate(snake):
            x, y = int(segment.x // self.block_size), int(segment.y // self.block_size)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y][x] = i + 1
        return grid

    def create_food_grid(self, food):
        grid = np.full((self.grid_size, self.grid_size), -1)
        if food:
            x, y = int(food.x // self.block_size), int(food.y // self.block_size)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y][x] = 1
        return grid

    def rotate_grid(self, grid, direction):
        if direction == Direction.UP:
            grid = np.rot90(grid, k=1)
        elif direction == Direction.RIGHT:
            grid = np.rot90(grid, k=3)
        elif direction == Direction.DOWN:
            grid = np.rot90(grid, k=2)
        elif direction == Direction.LEFT:
            grid = np.rot90(grid, k=1)
        return grid

    def center_grid(self, grid, head_pos):
        cx, cy = head_pos
        half = self.grid_size // 2

        grid = np.array(grid)

        if grid.ndim != 2:
            raise ValueError(f"Input grid must be a 2D array. Got {grid.ndim}")
        
        padded = np.pad(grid, pad_width=half, mode='constant', constant_values=-1)
        cx += half
        cy += half

        centered = padded[cy - half:cy + half + 1, cx - half:cx + half + 1]
        return centered
    
    def get_normalized_input(self, snake, food, direction):
        snake_grid = self.create_snake_grid(snake)
        food_grid = self.create_food_grid(food)
        rotated_snake = self.rotate_grid(snake_grid, direction)
        rotated_food = self.rotate_grid(food_grid, direction)

        head = snake[0]
        head_x, head_y = int(head.x // self.block_size), int(head.y // self.block_size)

        centered_snake = self.center_grid(rotated_snake, (head_x, head_y))
        centered_food = self.center_grid(rotated_food, (head_x, head_y))

        return {
            "snake_grid": centered_snake,
            "food_grid": centered_food
        }
