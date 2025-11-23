# train.py
import time
from snake_ai import SnakeAI
from neat import NEAT
from rl import ReinforcementTrainer

def run_neat_evolution(generations=100):
    """Run NEAT evolution to create initial AI"""
    print("=== NEAT Evolution Phase ===")
    snake_ai = SnakeAI()
    neat = NEAT(snake_ai.input_size, snake_ai.output_size, population_size=50)
    
    def fitness_function(genome):
        score, total_steps, four_left_turns, four_right_turns, over_25_same_dir_count = snake_ai.play_game(genome, render=False, training=True)
        fitness = snake_ai.calculate_fitness(score, total_steps, 0, 50 + score, four_left_turns, four_right_turns, over_25_same_dir_count)
        return fitness
    
    print(f"Starting NEAT evolution: {generations} generations")
    print(f"Population: {neat.population_size}")
    print("Max steps: 50 + apples_eaten (max 625)")
    
    start_time = time.time()
    
    for gen in range(generations):
        current_gen_best_fitness = neat.run_generation(fitness_function)
        current_best_score = int(current_gen_best_fitness / 1000)
        
        print(f"Gen {gen:3d}: Fitness = {current_gen_best_fitness:8.2f}")
        
        if neat.best_genome_overall and gen % 10 == 0:
            neat.save_best("best_snake_current.pkl")
    
    neat.save_best("best_snake_final.pkl")
    
    elapsed_time = time.time() - start_time
    print(f"NEAT evolution completed")
    print(f"Generations: {generations}")
    print(f"Best fitness: {neat.best_fitness:.2f}")
    print(f"Best score: {int(neat.best_fitness / 1000)}")
    print(f"Time: {elapsed_time:.2f}s")
    
    return neat.best_genome_overall

def run_rl_finetuning(genome, episodes=100):
    """Fine-tune NEAT AI with reinforcement learning"""
    print("\n=== RL Fine-tuning Phase ===")
    trainer = ReinforcementTrainer(
        genome=genome,
        learning_rate=0.01,
        discount_factor=0.95,
        exploration_rate=0.2
    )
    
    best_score = trainer.train(episodes=episodes, render=False, speed=1000)
    
    return trainer.genome, best_score

def main():
    
        
    neat_generations = 100
    rl_episodes = 100
        
    # Run NEAT evolution
    best_genome = run_neat_evolution(neat_generations)
        
    # Run RL fine-tuning
    final_genome, best_score = run_rl_finetuning(best_genome, rl_episodes)
        
        # Save final model
    with open("snake_ai_final.pkl", 'wb') as f:
        import pickle
        pickle.dump(final_genome, f)
        
    print(f"Training completed. Final model saved to 'snake_ai_final.pkl'")
    print(f"Best RL score: {best_score}")
        

if __name__ == "__main__":
    main()