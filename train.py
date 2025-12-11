import time
from snake_ai import SnakeAI
from neat import NEAT
from rl import ReinforcementTrainer
import logging
import logging.handlers
import matplotlib.pyplot as plt

logger = logging.getLogger("snake_train")
logger.setLevel(logging.INFO)

fh = logging.handlers.RotatingFileHandler("training.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

def plot_fitness(fitnesses, mean_fitnesses, save_path: str | None = None):
    plt.figure(figsize=(8, 5))
    plt.title("Snake Training - Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(fitnesses, label="Best Fitness")
    plt.plot(mean_fitnesses, label="Mean Fitness")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_scores(scores, mean_scores, save_path: str | None = None):
    plt.figure(figsize=(8, 5))
    plt.title("Snake Training - Score (Food Collected)")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.plot(scores, label="Best Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_time(times, save_path: str | None = None):
    plt.figure(figsize=(8, 5))
    plt.title("Training Time per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Time (seconds)")
    plt.plot(times, label="Time per Generation")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def run_neat_evolution(generations=2):
    """Run NEAT evolution to create initial AI"""
    logger.info("=== NEAT Evolution Phase ===")
    snake_ai = SnakeAI()
    neat = NEAT(snake_ai.input_size, snake_ai.output_size, population_size=50)
    
    def fitness_function(genome):
        score, total_steps, four_left_turns, four_right_turns, over_19_same_dir_count = snake_ai.play_game(genome, render=False, training=True)
        fitness = snake_ai.calculate_fitness(score, total_steps, 0, 50 + score, four_left_turns, four_right_turns, over_19_same_dir_count)
        return fitness, score
    
    logger.info(f"Starting NEAT evolution: {generations} generations")
    logger.info(f"Population: {neat.population_size}")
    logger.info(f"Input size: {snake_ai.input_size} (19x19 grid)")
    logger.info("Max steps: 50 + apples_eaten (max 361)")
    time_history = []
    start_time = time.time()
    
    for gen in range(generations):
        gen_start_time = time.time()
        current_gen_best_fitness = neat.run_generation(fitness_function)
        current_best_score = int(current_gen_best_fitness / 1000)
        gen_time = time.time() - gen_start_time  
        time_history.append(gen_time)
        
        if gen == 0 or gen % 10 == 0:
            plot_fitness(
                neat.fitness_history,
                neat.mean_fitness_history,
                save_path=f"./plots/fitness/fitness_{gen}.png")
            plot_scores(
                neat.score_history,
                neat.mean_score_history,
                save_path=f"./plots/score/score_{gen}.png")
            plot_time(time_history, save_path=f"./plots/time/time_{gen}.png")

        logger.info(f"Gen {gen:3d}: Fitness = {current_gen_best_fitness:8.2f} (Score ~{current_best_score}) | Time: {time.time() - gen_start_time:.2f}s")
        
        if neat.best_genome_overall and gen % 10 == 0:
            neat.save_best("best_snake_current.pkl")
    
    neat.save_best("best_snake_final.pkl")
    
    elapsed_time = time.time() - start_time
    logger.info(f"NEAT evolution completed")
    logger.info(f"Generations: {generations}")
    logger.info(f"Best fitness: {neat.best_fitness:.2f}")
    logger.info(f"Best score: {int(neat.best_fitness / 1000)}")
    logger.info(f"Time: {elapsed_time:.2f}s")
    
    return neat.best_genome_overall

def run_rl_finetuning(genome, episodes=100):
    """Fine-tune NEAT AI with reinforcement learning"""
    logger.info("\n=== RL Fine-tuning Phase ===")
    trainer = ReinforcementTrainer(
        genome=genome,
        learning_rate=0.01,
        discount_factor=0.95,
        exploration_rate=0.2
    )
    
    best_score = trainer.train(episodes=episodes, render=False, speed=1000)
    
    return trainer.genome, best_score

def main():
    neat_generations = 300
    rl_episodes = 10
        
    # Run NEAT evolution
    best_genome = run_neat_evolution(neat_generations)
        
    # Run RL fine-tuning
    final_genome, best_score = run_rl_finetuning(best_genome, rl_episodes)
        
    # Save final model
    with open("snake_ai_final.pkl", 'wb') as f:
        import pickle
        pickle.dump(final_genome, f)
        
    logger.info(f"Training completed. Final model saved to 'snake_ai_final.pkl'")
    logger.info(f"Best RL score: {best_score}")

if __name__ == "__main__":
    main()