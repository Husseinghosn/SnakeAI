import pygame
import sys
import torch
import numpy as np

from game import SnakeGame, GRID_SIZE
from snake_ai import SnakeAIStateBuilder
from rl import SnakeAgent

CELL_SIZE = 25
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
HUD_HEIGHT = 120
FPS = 12


# -------------------------------------------------------
# DRAWING UTILITIES
# -------------------------------------------------------
def draw_glass_panel(screen, width, height, opacity=170):
    """Glassy semi-transparent black panel."""
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, opacity))
    screen.blit(overlay, (0, 0))


def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    label = font.render(text, True, color)
    screen.blit(label, (x, y))


def draw_centered_text(screen, text, y, font, color=(255, 255, 255)):
    label = font.render(text, True, color)
    screen.blit(label, (WINDOW_SIZE // 2 - label.get_width() // 2, y))


def draw_grid(screen):
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (x, HUD_HEIGHT), (x, WINDOW_SIZE + HUD_HEIGHT))
    for y in range(HUD_HEIGHT, WINDOW_SIZE + HUD_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (0, y), (WINDOW_SIZE, y))


def draw_snake(screen, snake):
    for (x, y) in snake:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + HUD_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (0, 200, 0), rect, border_radius=6)


def draw_food(screen, food):
    fx, fy = food
    rect = pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE + HUD_HEIGHT, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (255, 70, 70), rect, border_radius=6)


# -------------------------------------------------------
# NEXT-LEVEL PREMIUM HUD
# -------------------------------------------------------
def draw_premium_hud(screen, stats, font_large, font_mid, font_small):
    draw_glass_panel(screen, WINDOW_SIZE, HUD_HEIGHT, opacity=165)

    # ================= BIG TOP LINE (Episode and Score) =================
    draw_centered_text(
        screen,
        f"EPISODE {stats['episode']}  |  SCORE: {stats['score']}  |  BEST: {stats['best']}",
        10,
        font_large,
        (0, 230, 255)
    )

    # ================= LEFT SIDE STATS =================
    left_x = 20
    y_start = 50

    left_stats = [
        f"Steps: {stats['steps']}",
        f"Snake Length: {stats['length']}",
        f"Head: {stats['head']}",
        f"Food: {stats['food']}",
    ]

    for i, item in enumerate(left_stats):
        draw_text(screen, item, left_x, y_start + i * 22, font_mid, (255, 255, 255))

    # ================= RIGHT SIDE STATS =================
    right_x = WINDOW_SIZE - 260
    right_stats = [
        f"Avg Q-Value: {stats['avg_q']:.3f}",
        f"Grid Size: {GRID_SIZE} x {GRID_SIZE}",
        f"Speed: {FPS} FPS",
    ]

    for i, item in enumerate(right_stats):
        draw_text(screen, item, right_x, y_start + i * 22, font_mid, (180, 180, 180))


# -------------------------------------------------------
# MAIN VISUAL LOOP
# -------------------------------------------------------
def run_visual_test(num_episodes=10):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + HUD_HEIGHT))
    pygame.display.set_caption("Snake AI — Premium Visual Gameplay")

    clock = pygame.time.Clock()

    # Fonts
    font_large = pygame.font.SysFont("Arial", 26, bold=True)
    font_mid = pygame.font.SysFont("Arial", 20)
    font_small = pygame.font.SysFont("Arial", 16)

    # Agent
    agent = SnakeAgent()
    agent.model.load_state_dict(torch.load("best_snake_dqn.pth"))
    agent.model.eval()
    agent.epsilon = 0.0

    state_builder = SnakeAIStateBuilder()
    best_score = 0

    for episode in range(1, num_episodes + 1):
        game = SnakeGame()
        game_state = game.reset()
        state = state_builder.build_state(game_state)

        done = False
        steps = 0
        score = 0
        q_values_list = []

        while not done:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_vals = agent.model(state_tensor)
            action = torch.argmax(q_vals).item()
            q_values_list.append(float(torch.max(q_vals).item()))

            game_state, reward, done, score = game.step(action)
            next_state = state_builder.build_state(game_state)
            state = next_state
            steps += 1
            best_score = max(best_score, score)

            # ---------------- DRAW EVERYTHING ----------------
            screen.fill((18, 18, 18))
            draw_grid(screen)
            draw_snake(screen, game.snake)
            draw_food(screen, game.food)

            stats = {
                "episode": episode,
                "score": score,
                "steps": steps,
                "length": len(game.snake),
                "head": tuple(game.snake[0]),
                "food": tuple(game.food),
                "avg_q": np.mean(q_values_list) if q_values_list else 0,
                "best": best_score
            }

            draw_premium_hud(screen, stats, font_large, font_mid, font_small)

            pygame.display.flip()

        print(f"Episode {episode} finished — Score: {score} | Steps: {steps}")

    pygame.quit()


if __name__ == "__main__":
    run_visual_test(10)
