import numpy as np
from game import GRID_SIZE, UP, DOWN, LEFT, RIGHT

class SnakeAIStateBuilder:

    def __init__(self):
        pass

    # ---------------------------------------------------
    # MAIN: Build 11-feature state vector
    # ---------------------------------------------------
    def build_state(self, state):
        grid     = state["grid"]
        snake    = state["snake"]
        food     = state["food"]
        direction = state["direction"]

        head_x, head_y = snake[0]

        # Neighbouring cells
        left_dir  = self.turn_left(direction)
        right_dir = self.turn_right(direction)
        straight  = direction

        # Danger flags
        danger_straight = 1 if self.is_collision(head_x, head_y, straight, snake) else 0
        danger_left     = 1 if self.is_collision(head_x, head_y, left_dir, snake) else 0
        danger_right    = 1 if self.is_collision(head_x, head_y, right_dir, snake) else 0

        # Food relative location
        food_left  = 1 if food[0] < head_x else 0
        food_right = 1 if food[0] > head_x else 0
        food_up    = 1 if food[1] < head_y else 0
        food_down  = 1 if food[1] > head_y else 0

        # Direction one-hot
        moving_left  = 1 if direction == LEFT else 0
        moving_right = 1 if direction == RIGHT else 0
        moving_up    = 1 if direction == UP else 0
        moving_down  = 1 if direction == DOWN else 0

        # Final compact state (11 values)
        state_vector = np.array([
            danger_straight,
            danger_left,
            danger_right,
            food_left,
            food_right,
            food_up,
            food_down,
            moving_left,
            moving_right,
            moving_up,
            moving_down
        ], dtype=int)

        return state_vector

    # ---------------------------------------------------
    # HELPERS
    # ---------------------------------------------------
    def is_collision(self, x, y, move_dir, snake):
        dx, dy = move_dir
        nx = x + dx
        ny = y + dy

        # wall hit
        if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
            return True

        # body hit
        if (nx, ny) in snake:
            return True

        return False

    # Turning utility
    def turn_left(self, d):
        if d == UP:    return LEFT
        if d == LEFT:  return DOWN
        if d == DOWN:  return RIGHT
        if d == RIGHT: return UP

    def turn_right(self, d):
        if d == UP:    return RIGHT
        if d == RIGHT: return DOWN
        if d == DOWN:  return LEFT
        if d == LEFT:  return UP
