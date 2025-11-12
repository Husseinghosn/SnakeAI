import numpy as np
from collections import namedtuple

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
        if direction.name == 'UP':
            grid = np.rot90(grid, k=0)
        elif direction.name == 'RIGHT':
            grid = np.rot90(grid, k=1)  
        elif direction.name == 'DOWN':
            grid = np.rot90(grid, k=2)
        elif direction.name == 'LEFT':
            grid = np.rot90(grid, k=3)  
        return grid
    
    def rotate_vector(self, dx, dy, direction):
        if direction.name == 'UP':
            return dx, dy
        elif direction.name == 'RIGHT':
            return dy, -dx 
        elif direction.name == 'DOWN':
            return -dx, -dy
        elif direction.name == 'LEFT':
            return -dy, dx  
        return dx, dy

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

        # Create grids in world coordinates
        snake_grid = self.create_snake_grid(snake)
        food_grid = self.create_food_grid(food)

        world_head_pos = np.where(snake_grid == 1)
        world_head_x = int(world_head_pos[1][0])  
        world_head_y = int(world_head_pos[0][0])  

        food_pos = np.where(food_grid == 1)
        if len(food_pos[0]) > 0:  # Food exists
            world_food_x = int(food_pos[1][0])
            world_food_y = int(food_pos[0][0])

            dx = world_food_x - world_head_x
            dy = world_food_y - world_head_y

            half_grid = self.grid_size // 2  

            if dx > half_grid:
                dx = dx - self.grid_size
            elif dx < -half_grid:  
                dx = dx + self.grid_size

            if dy > half_grid:
                dy = dy - self.grid_size
            elif dy < -half_grid:
                dy = dy + self.grid_size

            if not (-half_grid <= dx <= half_grid):
                print(f" WARNING: dx={dx} out of valid range [-{half_grid}, {half_grid}]")
            if not (-half_grid <= dy <= half_grid):
                print(f" WARNING: dy={dy} out of valid range [-{half_grid}, {half_grid}]")

            rotated_dx, rotated_dy = self.rotate_vector(dx, dy, direction)

            if not (-half_grid <= rotated_dx <= half_grid):
                print(f" WARNING: After rotation, rotated_dx={rotated_dx} out of valid range!")
                print(f"   Original (dx, dy) = ({dx}, {dy}), Direction = {direction.name}")
            if not (-half_grid <= rotated_dy <= half_grid):
                print(f" WARNING: After rotation, rotated_dy={rotated_dy} out of valid range!")
                print(f"   Original (dx, dy) = ({dx}, {dy}), Direction = {direction.name}")
        else:
            rotated_dx, rotated_dy = None, None

        rotated_snake = self.rotate_grid(snake_grid, direction)

        rotated_head_pos = np.where(rotated_snake == 1)
        rotated_head_x = int(rotated_head_pos[1][0])
        rotated_head_y = int(rotated_head_pos[0][0])

        centered_snake = self.center_grid(rotated_snake, (rotated_head_x, rotated_head_y))

        if rotated_dx is not None and rotated_dy is not None:
            centered_food = np.full((self.grid_size, self.grid_size), -1)
            food_centered_x = self.grid_size // 2 + rotated_dx
            food_centered_y = self.grid_size // 2 + rotated_dy

            if not (0 <= food_centered_x < self.grid_size and 0 <= food_centered_y < self.grid_size):
                print(f"FOOD OUT OF BOUNDS!")
                print(f"   World head: ({world_head_x}, {world_head_y})")
                print(f"   World food: ({world_food_x}, {world_food_y})")
                print(f"   World relative (dx, dy): ({dx}, {dy}) [after wrap]")
                print(f"   Direction: {direction}")
                print(f"   Rotated relative (rotated_dx, rotated_dy): ({rotated_dx}, {rotated_dy})")
                print(f"   Final position: ({food_centered_x}, {food_centered_y})")
                print(f"   Grid size: {self.grid_size}, Center: {self.grid_size // 2}")

            if 0 <= food_centered_x < self.grid_size and 0 <= food_centered_y < self.grid_size:
                centered_food[food_centered_y][food_centered_x] = 1
        else:
            centered_food = np.full((self.grid_size, self.grid_size), -1)

        return {"snake_grid": centered_snake, "food_grid": centered_food}
