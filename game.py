import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
HEAD_COLOR = (0, 255, 0)    
HEAD_COLOR2 = (0, 200, 0)   # new: head inner color (darker green)


BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
    
    def __init__(self, w=500, h=500):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        # self.score = 0
        # self.food = None
        # self._place_food()

        start_x = (self.w // 2) // BLOCK_SIZE * BLOCK_SIZE
        start_y = (self.h // 2) // BLOCK_SIZE * BLOCK_SIZE
        self.head = Point(start_x, start_y)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()

        self.pressed_direction = None
        
    def _opposite(self, d1, d2):
        return (d1 == Direction.LEFT and d2 == Direction.RIGHT) or \
               (d1 == Direction.RIGHT and d2 == Direction.LEFT) or \
               (d1 == Direction.UP and d2 == Direction.DOWN) or \
               (d1 == Direction.DOWN and d2 == Direction.UP)
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    new_dir = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    new_dir = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    new_dir = Direction.UP
                elif event.key == pygame.K_DOWN:
                    new_dir = Direction.DOWN
                else:
                    new_dir = None

                # set pressed_direction only if not reversing 180°
                if new_dir and not self._opposite(self.direction, new_dir):
                    self.pressed_direction = new_dir

            if event.type == pygame.KEYUP:
                # stop movement when released (only if releasing the current pressed key)
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    # if the released key corresponds to the pressed_direction, stop
                    key_to_dir = {
                        pygame.K_LEFT: Direction.LEFT,
                        pygame.K_RIGHT: Direction.RIGHT,
                        pygame.K_UP: Direction.UP,
                        pygame.K_DOWN: Direction.DOWN
                    }
                    released_dir = key_to_dir.get(event.key)
                    if released_dir == self.pressed_direction:
                        self.pressed_direction = None
        
        # If no key is pressed, don't move — just update UI and wait
        if not self.pressed_direction:
            self._update_ui()
            self.clock.tick(SPEED)
            return False, self.score

        # 2. move while key held (continuous movement)
        self.direction = self.pressed_direction
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
        # # 1. collect user input
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             self.direction = Direction.LEFT
        #         elif event.key == pygame.K_RIGHT:
        #             self.direction = Direction.RIGHT
        #         elif event.key == pygame.K_UP:
        #             self.direction = Direction.UP
        #         elif event.key == pygame.K_DOWN:
        #             self.direction = Direction.DOWN
        
        # # 2. move
        # self._move(self.direction) # update the head
        # self.snake.insert(0, self.head)
        
        # # 3. check if game over
        # game_over = False
        # if self._is_collision():
        #     game_over = True
        #     return game_over, self.score
            
        # # 4. place new food or just move
        # if self.head == self.food:
        #     self.score += 1
        #     self._place_food()
        # else:
        #     self.snake.pop()
        
        # # 5. update ui and clock
        # self._update_ui()
        # self.clock.tick(SPEED)
        # # 6. return game over and score
        # return game_over, self.score
    
    def _is_collision(self):
        # hits itself
        if self.head in self.snake[1:]:
            return True

        return False
    
    # def _is_collision(self):
    #     # hits boundary
    #     if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
    #         return True
    #     # hits itself
    #     if self.head in self.snake[1:]:
    #         return True
        
    #     return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for i, pt in enumerate(self.snake):
            if i == 0:
                outer = HEAD_COLOR
                inner = HEAD_COLOR2
            else:
                outer = BLUE1
                inner = BLUE2
            pygame.draw.rect(self.display, outer, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, inner, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        # x = self.head.x
        # y = self.head.y

        x = int(self.head.x)
        y = int(self.head.y)
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        # self.head = Point(x, y)

        if x < 0:
            x = self.w - BLOCK_SIZE
        elif x > self.w - BLOCK_SIZE:
            x = 0
        if y < 0:
            y = self.h - BLOCK_SIZE
        elif y > self.h - BLOCK_SIZE:
            y = 0

        self.head = Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        

    pygame.quit()

