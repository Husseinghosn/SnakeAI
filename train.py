# train.py
import time
import copy
from snake_ai import SnakeAI
from neat import NEAT
from rl import ReinforcementTrainer
import logging
import logging.handlers
logger = logging.getLogger("snake_train")
logger.setLevel(logging.INFO)

fh = logging.handlers.RotatingFileHandler("training.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

def run_lamarckian_evolution(generations=100, rl_episodes_per_genome=2):
    """Run NEAT evolution with Lamarckian weight updates via RL"""
    logger.info("=== Snake AI Training - Lamarckian Evolution ===")
    logger.info("=== NEAT Evolution Phase ===")
    snake_ai = SnakeAI()
    neat = NEAT(snake_ai.input_size, snake_ai.output_size, population_size=50)
    
    def fitness_function(genome):
        trainer = ReinforcementTrainer(
            genome=copy.deepcopy(genome),  
            learning_rate=0.02,
            discount_factor=0.9,
            exploration_rate=0.1
        )
        
        trainer.fine_tune_genome(episodes=rl_episodes_per_genome)
        
        for conn_key in genome.connections:
            genome.connections[conn_key].weight = trainer.genome.connections[conn_key].weight
        score, total_steps, four_left_turns, four_right_turns, over_25_same_dir_count = snake_ai.play_game(genome, render=False, training=True)
        fitness = snake_ai.calculate_fitness(score, total_steps, 0, 50 + score, four_left_turns, four_right_turns, over_25_same_dir_count)
        return fitness
    
    logger.info(f"Starting Lamarckian Evolution")
    logger.info(f"Generations: {generations}")
    logger.info(f"Population: {neat.population_size}")
    logger.info(f"RL episodes per genome: {rl_episodes_per_genome}")
    logger.info("Method: NEAT evolves structure + RL adjusts weights")
    
    start_time = time.time()
    
    for gen in range(generations):
        gen_start_time = time.time()
        current_gen_best_fitness = neat.run_generation(fitness_function)
        
        logger.info(f"Gen {gen:3d}: Fitness = {current_gen_best_fitness:8.2f} | Time: {time.time() - gen_start_time:.2f}s ")
        
        if neat.best_genome_overall and gen % 10 == 0:
            neat.save_best("best_snake_current.pkl")
    
    neat.save_best("best_snake_final.pkl")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed")
    logger.info(f"Generations: {generations}")
    logger.info(f"Best fitness: {neat.best_fitness:.2f}")
    logger.info(f"Time: {elapsed_time:.2f}s")
    logger.info(f"Saved: best_snake_final.pkl")

def main():
    generations = 100
    rl_episodes_per_genome = 2
    
    run_lamarckian_evolution(generations, rl_episodes_per_genome)
    
    logger.info(f"Training completed. Best genome saved to 'best_snake_final.pkl'")

if __name__ == "__main__":
    main()