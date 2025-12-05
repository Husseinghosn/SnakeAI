import time
import numpy as np
import torch
from tabulate import tabulate  # pip install tabulate

from game import SnakeGame
from snake_ai import SnakeAIStateBuilder
from rl import SnakeAgent


def format_header(title: str):
    print("\n" + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80)


def format_section(title: str):
    print("\n" + "-" * 80)
    print(f"{title}")
    print("-" * 80)


def test_dqn(num_episodes: int = 20, max_steps_per_episode: int = 2000):
    # Initialize agent
    agent = SnakeAgent()
    agent.model.load_state_dict(torch.load("best_snake_dqn.pth"))
    agent.model.eval()               # deterministic mode
    agent.epsilon = 0.0              # no randomness during testing

    state_builder = SnakeAIStateBuilder()

    format_header("SNAKE — DQN AGENT TESTING (FINAL REPORT MODE)")

    results = []
    all_scores = []
    all_steps = []

    start_time = time.time()

    for ep in range(1, num_episodes + 1):

        game = SnakeGame()
        game_state = game.reset()
        state = state_builder.build_state(game_state)

        done = False
        steps = 0
        score = 0
        q_max_values = []

        # Episode header
        print(f"\n[ Episode {ep} ]".center(80, "="))

        while not done and steps < max_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Forward pass
            q_values = agent.model(state_tensor)
            action = torch.argmax(q_values).item()
            q_max = torch.max(q_values).item()
            q_max_values.append(q_max)

            next_game_state, reward, done, score = game.step(action)
            next_state = state_builder.build_state(next_game_state)

            # Update state
            state = next_state
            steps += 1

        all_scores.append(score)
        all_steps.append(steps)

        # Episode summary
        results.append([
            ep,
            score,
            steps,
            round(np.mean(q_max_values), 3),
            game.snake[0][0],              # head x
            game.snake[0][1],              # head y
            len(game.snake),               # snake length
            game.food[0],                  # food x
            game.food[1],                  # food y
        ])

        # Print in-console stats for each episode
        print(tabulate(
            [["Score", score],
             ["Steps Survived", steps],
             ["Final Snake Length", len(game.snake)],
             ["Avg Q-value", round(np.mean(q_max_values), 3)],
             ["Head Position", f"({game.snake[0][0]}, {game.snake[0][1]})"],
             ["Food Position", f"({game.food[0]}, {game.food[1]})"]],
            headers=["Metric", "Value"],
            tablefmt="fancy_grid"
        ))

    # ---------------------------
    # Final Testing Summary
    # ---------------------------
    format_section("FINAL SUMMARY TABLE")

    headers = [
        "Ep", "Score", "Steps",
        "Avg Q", "HeadX", "HeadY",
        "Length", "FoodX", "FoodY"
    ]

    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

    avg_score = np.mean(all_scores)
    avg_steps = np.mean(all_steps)
    best_score = np.max(all_scores)
    done_time = time.time() - start_time

    # Final stats box
    format_header("OVERALL PERFORMANCE REPORT")

    print(tabulate(
        [
            ["Episodes Tested", num_episodes],
            ["Average Score", round(avg_score, 2)],
            ["Average Steps", round(avg_steps, 2)],
            ["Best Score", best_score],
            ["Total Time (s)", round(done_time, 2)],
            ["Total Time (min)", round(done_time / 60, 2)],
        ],
        headers=["Metric", "Value"],
        tablefmt="fancy_grid"
    ))

    print("\nTesting completed successfully!\n")


def main():
    test_dqn(num_episodes=20)


if __name__ == "__main__":
    main()
