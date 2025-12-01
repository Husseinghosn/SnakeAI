import pygame
import numpy as np

class GridView:
    def __init__(self):
        self.grid_size = 19
        self.cell_size = 15  # Each cell is 15 pixels
        self.width = self.cell_size * self.grid_size + 20
        self.height = self.cell_size * self.grid_size + 80  # More space for info

        self.surface = pygame.Surface((self.width, self.height))

        # Colors
        self.bg_color = (0, 0, 0)
        self.grid_color = (50, 50, 50)
        self.snake_head_color = (0, 255, 0)
        self.snake_body_start = (0, 200, 0)
        self.food_color = (255, 0, 0)
        self.empty_color = (20, 20, 20)
        self.text_color = (255, 255, 255)
        self.apple_coord_color = (255, 255, 0)  # Yellow for apple coordinates
        
        try:
            self.title_font = pygame.font.Font('arial.ttf', 14)
            self.cell_font = pygame.font.Font('arial.ttf', 8)
            self.info_font = pygame.font.Font('arial.ttf', 10)
        except:
            self.title_font = pygame.font.SysFont('arial', 14)
            self.cell_font = pygame.font.SysFont('arial', 8)
            self.info_font = pygame.font.SysFont('arial', 10)

    def get_color_for_value(self, value):
        """Get color based on cell value"""
        if value == -1:  # Food
            return self.food_color
        elif value == 0:  # Empty
            return self.empty_color
        elif -1 < value < 0 or 0 < value:  # Body segments (normalized) or apple coordinates
            # Body segments are normalized to [0, 1], apple coords are in [-1, 1]
            if value > 0:  # Body segment
                # Green gradient based on normalized value
                intensity = int(50 + value * 205)  # 50-255 range
                return (0, intensity, 0)
            else:  # Apple coordinate (negative but not -1)
                # Yellow for apple coordinates
                intensity = int(150 + abs(value) * 105)  # 150-255 range
                return (intensity, intensity, 0)
        else:
            # Edge cases - use default
            return self.empty_color

    def draw_grid(self, grid):
        """Draw the merged grid"""
        # Find the head and body positions
        center = self.grid_size // 2
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                value = grid[y][x]
                color = self.get_color_for_value(value)
                
                rect = pygame.Rect(
                    x * self.cell_size + 10,  # 10px padding
                    y * self.cell_size + 60,  # 60px for title and info
                    self.cell_size - 1,
                    self.cell_size - 1
                )
                
                pygame.draw.rect(self.surface, color, rect)
                
                # Add special highlighting
                if y == center and x == center:
                    # Head position (now contains apple X coordinate)
                    pygame.draw.rect(self.surface, (255, 255, 255), rect, 2)
                    
                    # Draw apple X coordinate
                    text = self.cell_font.render(f"X:{value:.2f}", True, self.text_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.surface.blit(text, text_rect)
                    
                elif value == -1:  # Food
                    inner_rect = rect.inflate(-3, -3)
                    pygame.draw.rect(self.surface, (255, 150, 150), inner_rect)
                    
                    # Draw food marker
                    text = self.cell_font.render("F", True, self.text_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.surface.blit(text, text_rect)
                    
                elif -1 < value < 0:  # Apple Y coordinate (at body behind head)
                    # Highlight cell with apple Y coordinate
                    pygame.draw.rect(self.surface, (255, 200, 0), rect, 2)
                    
                    # Draw apple Y coordinate
                    text = self.cell_font.render(f"Y:{value:.2f}", True, self.text_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.surface.blit(text, text_rect)
                    
                elif 0 < value:  # Body segment
                    # Draw normalized body value
                    text = self.cell_font.render(f"{value:.1f}", True, self.text_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.surface.blit(text, text_rect)
                
                # Draw grid lines
                pygame.draw.rect(self.surface, self.grid_color, rect, 1)

    def update(self, grid):
        self.surface.fill(self.bg_color)
        
        # Draw title
        title = self.title_font.render("AI Input Grid (19x19)", True, self.text_color)
        self.surface.blit(title, (10, 5))
        
        
        
        # Find apple coordinates
        center = self.grid_size // 2
        apple_x = grid[center, center] if 0 <= center < self.grid_size else 0
        apple_y = 0
        
        # Try to find apple Y coordinate (will be where body value 2 used to be)
        # It could be at various positions, so we'll search for a negative value that's not -1
        apple_y_positions = np.where((grid < 0) & (grid > -1))
        if len(apple_y_positions[0]) > 0:
            y_idx = apple_y_positions[0][0]
            x_idx = apple_y_positions[1][0]
            apple_y = grid[y_idx, x_idx]
        
        # Draw apple coordinate info
        coord_text = self.info_font.render(f"Apple relative: X={apple_x:.2f}, Y={apple_y:.2f}", 
                                          True, (255, 255, 0))
        self.surface.blit(coord_text, (10, self.height - 20))
        
        # Draw the grid
        self.draw_grid(grid)
        
        return self.surface