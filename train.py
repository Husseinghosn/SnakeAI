import time
import logging
import logging.handlers
from collections import deque

import matplotlib.pyplot as plt

from game import SnakeGame
from snake_ai import SnakeAIStateBuilder
from rl import SnakeAgent


logger = logging.getLogger("snake_train_dqn")
logger.setLevel(logging.INFO)

fh = logging.handlers.RotatingFileHandler(
    "training_dqn.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
)
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(ch)


def plot_scores(scores, mean_scores, save_path: str | None = None):
    plt.figure()
    plt.title("Snake DQN Training")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def train_dqn(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 2000,
    plot_every: int = 50,
):
    game = SnakeGame()
    state_builder = SnakeAIStateBuilder()
    agent = SnakeAgent()

    scores = []
    mean_scores = []
    score_window = deque(maxlen=100)

    best_score = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        game_state = game.reset()
        state = state_builder.build_state(game_state)
        done = False
        episode_score = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # choose action
            action = agent.act(state)

            # step environment
            next_game_state, reward, done, score = game.step(action)
            next_state = state_builder.build_state(next_game_state)

            # store + learn
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            episode_score = score
            steps += 1

        scores.append(episode_score)
        score_window.append(episode_score)
        mean_score = sum(score_window) / len(score_window)
        mean_scores.append(mean_score)

        # Save best model
        if episode_score > best_score:
            best_score = episode_score
            import torch
            torch.save(agent.model.state_dict(), "best_snake_dqn.pth")

        # Log status
        if episode % plot_every == 0 or episode == 1:
            elapsed = time.time() - start_time
            logger.info(
                f"Ep {episode:4d} | Score: {episode_score:3d} | "
                f"Best: {best_score:3d} | Mean(100): {mean_score:5.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | Time: {elapsed:.1f}s"
            )
            plot_scores(scores, mean_scores, save_path="training_dqn_curve.png")

    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time:.1f} seconds.")
    logger.info(f"Best score achieved: {best_score}")
    logger.info("Saved curve: training_dqn_curve.png")


def main():
    train_dqn(num_episodes=1000)


if __name__ == "__main__":
    main()
