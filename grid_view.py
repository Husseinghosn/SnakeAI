import pygame
import numpy as np

class GridView:
    def __init__(self, width=250, height=250, grid_size=15):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cell_size = width // grid_size
        
        # # Create surface instead of window
        self.surface = pygame.Surface((width , height* 2 + 10))

        
        # Colors
        self.bg_color = (0, 0, 0)
        self.grid_color = (50, 50, 50)
        self.snake_colors = [(0, 255, 0), (0, 200, 0)]
        self.food_color = (255, 0, 0)
        self.empty_color = (20, 20, 20)
        self.text_color = (255, 255, 255)
        
        try:
            self.font = pygame.font.Font('arial.ttf', 10)
        except:
            self.font = pygame.font.SysFont('arial', 10)

    def draw_grid(self, grid, offset_x=0, offset_y=0, is_food_grid=False):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                value = grid[y][x]
                rect = pygame.Rect(
                    offset_x + x * self.cell_size,
                    offset_y + y * self.cell_size,
                    self.cell_size - 1,
                    self.cell_size - 1
                )

                # 🎨 Color logic
                if is_food_grid:
                    color = self.food_color if value == 1 else self.empty_color
                elif value == -1:
                    color = self.empty_color
                elif value == 1:
                    color = self.snake_colors[0]  # Head (bright green)
                elif value > 1:
                    # Fade green based on segment index
                    fade = max(50, 255 - (value - 1) * 10)
                    color = (0, fade, 0)
                else:
                    color = self.empty_color  # fallback

                # 🧱 Draw cell background
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, self.grid_color, rect, 1)

                # 🔢 Always draw the number
                text = self.font.render(str(value), True, self.text_color)
                text_rect = text.get_rect(center=(
                    offset_x + x * self.cell_size + self.cell_size // 2,
                    offset_y + y * self.cell_size + self.cell_size // 2
                ))
                self.surface.blit(text, text_rect)


    # def draw_grid(self, grid, offset_x=0, offset_y=0, is_food_grid=False):
    #     for y in range(self.grid_size):
    #         for x in range(self.grid_size):
    #             value = grid[y][x]
    #             rect = pygame.Rect(
    #                 offset_x + x * self.cell_size,
    #                 offset_y + y * self.cell_size,
    #                 self.cell_size - 1,
    #                 self.cell_size - 1
    #             )
                
    #             # Set cell color
    #             # if value == -1:
    #             #     color = self.empty_color
    #             # elif is_food_grid:
    #             #     color = self.food_color if value == 1 else self.empty_color
    #             # else:  # Snake grid
    #             #     color = self.snake_colors[0] if value == 1 else self.snake_colors[1]
    #             # if is_food_grid:
    #             #     color = self.food_color if value == 1 else self.empty_color
    #             # elif value == -1:
    #             #     color = self.empty_color
    #             # else:
    #             #     color = self.snake_colors[0] if value == 1 else self.snake_colors[1]

    #             if is_food_grid:
    #                 color = self.food_color if value == 1 else self.empty_color
    #             elif value == -1:
    #                 color = self.empty_color
    #             elif value == 1:
    #                 color = self.snake_colors[0]  # Head
    #             else:
    #                 # Fade color based on segment index
    #                 fade = max(50, 255 - value * 10)
    #                 color = (0, fade, 0)  # Darker green for tail


    #             # Draw cell background
    #             pygame.draw.rect(self.surface, color, rect)
    #             pygame.draw.rect(self.surface, self.grid_color, rect, 1)
                
    #             # Draw number for all cells (including -1)
    #             text = self.font.render(str(value), True, self.text_color)
    #             text_rect = text.get_rect(center=(
    #                 offset_x + x * self.cell_size + self.cell_size // 2,
    #                 y * self.cell_size + self.cell_size // 2
    #             ))
    #             self.surface.blit(text, text_rect)

    def create_snake_grid(self, snake):
        grid = np.full((self.grid_size, self.grid_size), -1)
        for i, segment in enumerate(snake):
            grid_x = int(segment.x // 20)
            grid_y = int(segment.y // 20)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_y][grid_x] = i + 1  # Positive increasing values
        return grid

    def create_food_grid(self, food):
        grid = np.full((self.grid_size, self.grid_size), -1)
        if food:
            food_x = int(food.x // 20)
            food_y = int(food.y // 20)
            if 0 <= food_x < self.grid_size and 0 <= food_y < self.grid_size:
                grid[food_y][food_x] = 1
        return grid

    # def update(self, game_state):
    #     self.surface.fill(self.bg_color)
        
    #     # Create and draw snake grid
    #     snake_grid = self.create_snake_grid(game_state.snake)
    #     self.draw_grid(snake_grid, 0, False)
        
    #     # Create and draw food grid
    #     food_grid = self.create_food_grid(game_state.food)
    #     self.draw_grid(food_grid, self.width + 10, True)
        
    #     # Draw labels
    #     snake_label = self.font.render("Snake Body", True, self.text_color)
    #     food_label = self.font.render("Apple", True, self.text_color)
    #     self.surface.blit(snake_label, (50, 5))
    #     self.surface.blit(food_label, (self.width + 60, 5))
        
    #     return self.surface

    def update(self, game_state):
        self.surface.fill(self.bg_color)

        # Snake grid at the top
        snake_grid = self.create_snake_grid(game_state.snake)
        self.draw_grid(snake_grid, offset_x=0, offset_y=0, is_food_grid=False)

        # Food grid below
        food_grid = self.create_food_grid(game_state.food)
        self.draw_grid(food_grid, offset_x=0, offset_y=self.height + 10, is_food_grid=True)

        # Labels
        snake_label = self.font.render("Snake Body", True, self.text_color)
        food_label = self.font.render("Apple", True, self.text_color)
        self.surface.blit(snake_label, (10, 5))
        self.surface.blit(food_label, (10, self.height + 15))

        return self.surface
