import torch
import numpy as np
from AI import SnakeAI, GeneticTrainer
from game import SnakeGame
import os

class AITrainer:
    def __init__(self, grid_size=25, population_size=30):
        self.grid_size = grid_size
        self.population_size = population_size
        
    def train_genetic_algorithm(self, generations=50):
        """Train using Genetic Algorithm"""
        print("=== Starting Genetic Algorithm Training ===")
        
        # Initialize game and GA trainer
        game = SnakeGame(w=500, h=500)
        ga_trainer = GeneticTrainer(
            population_size=self.population_size,
            grid_size=self.grid_size,
            mutation_rate=0.15
        )
        
        best_fitness = 0
        best_individual = None
        
        for gen in range(generations):
            print(f"GA Generation {gen + 1}/{generations}")
            
            # Evaluate population
            gen_fitness = []
            for i, individual in enumerate(ga_trainer.population):
                fitness = ga_trainer.evaluate_fitness(individual, game)
                gen_fitness.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual
                    print(f"  New best fitness: {best_fitness:.2f}")
                    
                    # Save best model
                    torch.save(best_individual.state_dict(), "best_snake_ai.pth")
            
            ga_trainer.fitness_scores = gen_fitness
            
            # Create next generation (except last generation)
            if gen < generations - 1:
                ga_trainer.create_new_generation()
            
            # Statistics
            avg_fitness = np.mean(gen_fitness)
            max_fitness = np.max(gen_fitness)
            print(f"  Avg: {avg_fitness:.2f}, Max: {max_fitness:.2f}")
            print("-" * 40)
        
        return best_individual

def load_trained_model(model_path="best_snake_ai.pth"):
    """Load a trained model"""
    model = SnakeAI()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Loaded trained model!")
    else:
        print("No trained model found, using random weights.")
    return model

if __name__ == "__main__":
    trainer = AITrainer(population_size=20)
    
    # Train with GA
    best_ai = trainer.train_genetic_algorithm(generations=30)
    
    print("Training completed! Best model saved as 'best_snake_ai.pth'")