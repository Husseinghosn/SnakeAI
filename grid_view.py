import pygame
import numpy as np

class GridView:
    def __init__(self):
        self.grid_size = 19
        self.cell_size = 15  # Each cell is 15 pixels
        self.width = self.cell_size * self.grid_size + 20
        self.height = self.cell_size * self.grid_size + 40

        self.surface = pygame.Surface((self.width, self.height))

        # Colors
        self.bg_color = (0, 0, 0)
        self.grid_color = (50, 50, 50)
        self.snake_head_color = (0, 255, 0)
        self.snake_body_start = (0, 200, 0)
        self.food_color = (255, 0, 0)
        self.empty_color = (20, 20, 20)
        self.text_color = (255, 255, 255)
        
        try:
            self.title_font = pygame.font.Font('arial.ttf', 14)
            self.cell_font = pygame.font.Font('arial.ttf', 8)
        except:
            self.title_font = pygame.font.SysFont('arial', 14)
            self.cell_font = pygame.font.SysFont('arial', 8)

    def get_color_for_value(self, value):
        """Get color based on cell value"""
        if value == -1:  # Food
            return self.food_color
        elif value == 0:  # Empty
            return self.empty_color
        elif value == 1:  # Snake head
            return self.snake_head_color
        elif value > 1:  # Snake body
            # Gradient based on distance from head
            fade = max(50, 255 - (value - 1) * 20)
            return (0, fade, 0)
        return self.empty_color

    def draw_grid(self, grid):
        """Draw the merged grid"""
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                value = grid[y][x]
                color = self.get_color_for_value(value)
                
                rect = pygame.Rect(
                    x * self.cell_size + 10,  # 10px padding
                    y * self.cell_size + 30,  # 30px for title
                    self.cell_size - 1,
                    self.cell_size - 1
                )
                
                pygame.draw.rect(self.surface, color, rect)
                
                # Add inner highlight for head and food
                if value == 1 or value == -1:
                    inner_rect = rect.inflate(-3, -3)
                    inner_color = (255, 150, 150) if value == -1 else (0, 150, 0)
                    pygame.draw.rect(self.surface, inner_color, inner_rect)
                
                # Draw grid lines
                pygame.draw.rect(self.surface, self.grid_color, rect, 1)
                
                # Draw value numbers for non-empty cells
                if value != 0:
                    text = self.cell_font.render(str(value), True, self.text_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.surface.blit(text, text_rect)

    def update(self, grid):
        self.surface.fill(self.bg_color)
        
        # Draw title
        title = self.title_font.render("Game Grid (19x19)", True, self.text_color)
        self.surface.blit(title, (10, 5))
        
        # Draw legend
        legend = self.cell_font.render("-1=Fruit, 1=Head, >1=Body", True, (200, 200, 200))
        self.surface.blit(legend, (10, 25))
        
        # Draw the grid
        self.draw_grid(grid)
        
        return self.surface