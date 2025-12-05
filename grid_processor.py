import numpy as np
from collections import namedtuple

Point = namedtuple('Point', 'x, y')

class GridProcessor:
    def __init__(self, grid_size=19, block_size=20):
        self.grid_size = grid_size
        self.block_size = block_size

    def create_merged_grid(self, snake, food):
        """Create a single grid with snake body and food"""
        grid = np.full((self.grid_size, self.grid_size), 0)  # 0 for empty space
        
        # Add snake body (head = 1, body segments increase by 1)
        for i, segment in enumerate(snake):
            grid_x = int(segment.x // self.block_size)
            grid_y = int(segment.y // self.block_size)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_y][grid_x] = i + 1
        
        # Add food (-1)
        if food:
            food_x = int(food.x // self.block_size)
            food_y = int(food.y // self.block_size)
            if 0 <= food_x < self.grid_size and 0 <= food_y < self.grid_size:
                # Only place food if cell is empty (not part of snake)
                if grid[food_y][food_x] == 0:
                    grid[food_y][food_x] = -1
        
        return grid

    def rotate_grid(self, grid, direction):
        """Rotate grid so snake always appears to be moving 'up' relative to network"""
        if direction.name == 'UP':
            return grid  # No rotation needed
        elif direction.name == 'RIGHT':
            return np.rot90(grid, k=1)  # Rotate 90° counter-clockwise (so right becomes up)
        elif direction.name == 'DOWN':
            return np.rot90(grid, k=2)  # Rotate 180°
        elif direction.name == 'LEFT':
            return np.rot90(grid, k=3)  # Rotate 90° clockwise (so left becomes up)
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

    def get_relative_apple_coords(self, grid):
        """Calculate relative apple coordinates from head"""
        # Find head position (should be at center after rotation and centering)
        head_pos = np.where(grid == 1)
        if len(head_pos[0]) == 0:
            return 0, 0  # No head found
        
        head_y, head_x = head_pos[0][0], head_pos[1][0]
        
        # Find apple position
        apple_pos = np.where(grid == -1)
        if len(apple_pos[0]) == 0:
            return 0, 0  # No apple found
        
        apple_y, apple_x = apple_pos[0][0], apple_pos[1][0]
        
        # Calculate relative coordinates
        rel_x = apple_x - head_x
        rel_y = apple_y - head_y
        
        # Normalize to -1 to 1 range
        max_dist = self.grid_size // 2  # Maximum possible distance in grid
        norm_rel_x = rel_x / max_dist if max_dist > 0 else 0
        norm_rel_y = rel_y / max_dist if max_dist > 0 else 0
        
        # Clamp to [-1, 1] in case of edge cases
        norm_rel_x = max(-1.0, min(1.0, norm_rel_x))
        norm_rel_y = max(-1.0, min(1.0, norm_rel_y))
        
        return norm_rel_x, norm_rel_y

    def normalize_body_values(self, grid):
        """Normalize snake body values (>=3) to range [0, 1]"""
        # Create a copy to avoid modifying the original
        normalized_grid = grid.copy().astype(np.float32)
        
        # Find body cells (values >= 3)
        body_mask = grid >= 1
        
        if np.any(body_mask):
            # Get the min and max body values
            body_values = grid[body_mask]
            min_body = np.min(body_values)
            max_body = np.max(body_values)
            
            # Normalize to [0, 1] range
            if max_body > min_body:
                normalized_grid[body_mask] = (body_values - min_body) / (max_body - min_body)
            else:
                # All body values are the same (edge case)
                normalized_grid[body_mask] = 0.5
        
        # Keep food (-1), empty (0), head (1), and body behind head (2) as they are for now
        return normalized_grid

    def get_normalized_input(self, snake, food, direction):
        """Create normalized input grid for neural network with apple coords"""
        # Create single merged grid
        merged_grid = self.create_merged_grid(snake, food)
        
        # Rotate based on snake's direction
        rotated_grid = self.rotate_grid(merged_grid, direction)
        
        # Center on snake head
        centered_grid = self.center_grid(rotated_grid)
        
        # Normalize snake body values (>=3) to [0, 1] range
        normalized_grid = self.normalize_body_values(centered_grid)
        
        # Get normalized relative apple coordinates
        apple_rel_x, apple_rel_y = self.get_relative_apple_coords(centered_grid)
        
        # Find head position in centered grid (should be at center)
        center = self.grid_size // 2
        head_y, head_x = center, center
        
        # Find the cell behind head (value = 2)
        behind_head_y, behind_head_x = center, center  # Initialize with center
        
        # Try to find where the body segment with value 2 is
        body_positions = np.where(centered_grid == 2)
        if len(body_positions[0]) > 0:
            behind_head_y, behind_head_x = body_positions[0][0], body_positions[1][0]
        
        # Replace head cell (value 1) with relative apple X coordinate
        normalized_grid[head_y, head_x] = apple_rel_x
        
        # Replace cell behind head (value 2) with relative apple Y coordinate
        normalized_grid[behind_head_y, behind_head_x] = apple_rel_y
        
        return {"grid": normalized_grid}