# train.py
import time
import copy
from snake_ai import SnakeAI
from neat import NEAT
from rl import ReinforcementTrainer
import logging
import logging.handlers

<<<<<<< HEAD
def run_lamarckian_evolution(generations=100, rl_episodes_per_genome=2):
    """Run NEAT evolution with Lamarckian weight updates via RL"""
    print("=== Snake AI Training - Lamarckian Evolution ===")
=======
logger = logging.getLogger("snake_train")
logger.setLevel(logging.INFO)

fh = logging.handlers.RotatingFileHandler("training.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

def run_neat_evolution(generations=2):
    """Run NEAT evolution to create initial AI"""
    logger.info("=== NEAT Evolution Phase ===")
>>>>>>> Type-1-AI
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
    
<<<<<<< HEAD
    print(f"Starting Lamarckian Evolution")
    print(f"Generations: {generations}")
    print(f"Population: {neat.population_size}")
    print(f"RL episodes per genome: {rl_episodes_per_genome}")
    print("Method: NEAT evolves structure + RL adjusts weights")
=======
    logger.info(f"Starting NEAT evolution: {generations} generations")
    logger.info(f"Population: {neat.population_size}")
    logger.info("Max steps: 50 + apples_eaten (max 361)")
>>>>>>> Type-1-AI
    
    start_time = time.time()
    
    for gen in range(generations):
        gen_start_time = time.time()
        current_gen_best_fitness = neat.run_generation(fitness_function)
        
        logger.info(f"Gen {gen:3d}: Fitness = {current_gen_best_fitness:8.2f} | Time: {time.time() - gen_start_time:.2f}s ")
        
        if neat.best_genome_overall and gen % 10 == 0:
            neat.save_best("best_snake_current.pkl")
    
    neat.save_best("best_snake_final.pkl")
    
    elapsed_time = time.time() - start_time
<<<<<<< HEAD
    print(f"Training completed")
    print(f"Generations: {generations}")
    print(f"Best fitness: {neat.best_fitness:.2f}")
    print(f"Time: {elapsed_time:.2f}s")
    print(f"Saved: best_snake_final.pkl")
=======
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
>>>>>>> Type-1-AI

def main():
    generations = 100
    rl_episodes_per_genome = 2
    
<<<<<<< HEAD
    run_lamarckian_evolution(generations, rl_episodes_per_genome)
    
    print(f"Training completed. Best genome saved to 'best_snake_final.pkl'")
=======
        
    neat_generations = 200
    rl_episodes = 100
        
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
        
>>>>>>> Type-1-AI

if __name__ == "__main__":
    main()