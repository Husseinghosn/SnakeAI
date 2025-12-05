import random
import numpy as np

GRID_SIZE = 20

CELL_EMPTY = 0
CELL_FOOD = 1
CELL_SNAKE = -1
CELL_WALL = -2

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class SnakeGame:
    def __init__(self):
        self.reset()

    # ---------------------------------------------------
    # RESET GAME
    # ---------------------------------------------------
    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0
        self.game_over = False
        self.steps = 0  # for survival streak bonus

        self.food = self.generate_food()
        return self.get_state()

    # ---------------------------------------------------
    # FOOD PLACEMENT
    # ---------------------------------------------------
    def generate_food(self):
        while True:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            if (x, y) not in self.snake:
                return (x, y)

    # ---------------------------------------------------
    # MAKING TURNS
    # ---------------------------------------------------
    def compute_new_direction(self, current_dir, action):
        dirs = [UP, RIGHT, DOWN, LEFT]
        idx = dirs.index(current_dir)

        # action: 0 straight, 1 left, 2 right
        if action == 1:       # left
            idx = (idx - 1) % 4
        elif action == 2:     # right
            idx = (idx + 1) % 4

        return dirs[idx]

    # ---------------------------------------------------
    # STEP LOGIC WITH REWARD SHAPING
    # ---------------------------------------------------
    def step(self, action):

        # 1. Base reward
        reward = 0.1  # survival reward

        # Penalty for turning unnecessarily
        if action != 0:
            reward -= 0.05

        # 2. Calculate old distance BEFORE the move
        head_x, head_y = self.snake[0]
        old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])

        # 3. Apply turn logic
        self.direction = self.compute_new_direction(self.direction, action)
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # 4. Collision check
        if (
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in self.snake
        ):
            self.game_over = True
            return self.get_state(), -10, True, self.score

        # 5. Move snake
        self.snake.insert(0, new_head)

        # 6. Check if food eaten
        if new_head == self.food:
            self.score += 1
            reward += 10
            self.food = self.generate_food()
        else:
            self.snake.pop()

        # 7. Distance-to-food reward shaping (after move)
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_distance < old_distance:
            reward += 0.2
        else:
            reward -= 0.2

        # 8. Survival streak bonus
        self.steps += 1
        if self.steps % 100 == 0:
            reward += 0.5

        return self.get_state(), reward, False, self.score

    # ---------------------------------------------------
    # BUILD STATE FOR AGENT
    # ---------------------------------------------------
    def get_state(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        # snake on grid
        for (x, y) in self.snake:
            grid[y][x] = CELL_SNAKE

        # food on grid
        fx, fy = self.food
        grid[fy][fx] = CELL_FOOD

        return {
            "grid": grid,
            "snake": self.snake,
            "food": self.food,
            "direction": self.direction
        }
