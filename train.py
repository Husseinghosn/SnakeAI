# train.py
import time
import copy
from snake_ai import SnakeAI
from neat import NEAT
from rl import ReinforcementTrainer

def run_lamarckian_evolution(generations=100, rl_episodes_per_genome=2):
    """Run NEAT evolution with Lamarckian weight updates via RL"""
    print("=== Snake AI Training - Lamarckian Evolution ===")
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
    
    print(f"Starting Lamarckian Evolution")
    print(f"Generations: {generations}")
    print(f"Population: {neat.population_size}")
    print(f"RL episodes per genome: {rl_episodes_per_genome}")
    print("Method: NEAT evolves structure + RL adjusts weights")
    
    start_time = time.time()
    
    for gen in range(generations):
        current_gen_best_fitness = neat.run_generation(fitness_function)
        
        print(f"Gen {gen:3d}: Fitness = {current_gen_best_fitness:8.2f}")
        
        if neat.best_genome_overall and gen % 10 == 0:
            neat.save_best("best_snake_current.pkl")
    
    neat.save_best("best_snake_final.pkl")
    
    elapsed_time = time.time() - start_time
    print(f"Training completed")
    print(f"Generations: {generations}")
    print(f"Best fitness: {neat.best_fitness:.2f}")
    print(f"Time: {elapsed_time:.2f}s")
    print(f"Saved: best_snake_final.pkl")

def main():
    generations = 100
    rl_episodes_per_genome = 2
    
    run_lamarckian_evolution(generations, rl_episodes_per_genome)
    
    print(f"Training completed. Best genome saved to 'best_snake_final.pkl'")

if __name__ == "__main__":
    main()