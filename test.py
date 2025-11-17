# test.py
import pygame
import numpy as np
from snake_ai import SnakeAI

def load_genome(filename):
    """Load a trained genome from file"""
    try:
        with open(filename, 'rb') as f:
            import pickle
            genome = pickle.load(f)
        print(f"Loaded genome: {filename}")
        print(f"Fitness: {genome.fitness:.2f}")
        return genome
    except Exception as e:
        print(f"Failed to load genome: {e}")
        return None

def test_ai(genome, num_games=5, speed=50, render=True):
    """Test the AI by playing multiple games"""
    snake_ai = SnakeAI()
    
    scores = []
    steps_per_game = []
    
    print(f"Testing AI for {num_games} games")
    
    for game_num in range(num_games):
        score, total_steps = snake_ai.play_game(genome, render=render, speed=speed)
        scores.append(score)
        steps_per_game.append(total_steps)
        efficiency = total_steps / score if score > 0 else total_steps
        print(f"Game {game_num + 1}: Score={score}, Steps={total_steps}, Eff={efficiency:.2f}")
    
    avg_score = np.mean(scores)
    avg_steps = np.mean(steps_per_game)
    avg_efficiency = avg_steps / avg_score if avg_score > 0 else avg_steps
    
    print(f"\nTest Results:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Efficiency: {avg_efficiency:.2f} steps/food")
    
    return scores

def analyze_genome(genome):
    """Analyze the structure of a genome"""
    print("\nGenome Analysis:")
    print(f"Nodes: {len(genome.nodes)}")
    print(f"Connections: {len(genome.connections)}")
    
    input_nodes = sum(1 for node in genome.nodes.values() if node.type == 'input')
    hidden_nodes = sum(1 for node in genome.nodes.values() if node.type == 'hidden')
    output_nodes = sum(1 for node in genome.nodes.values() if node.type == 'output')
    
    print(f"Input: {input_nodes}, Hidden: {hidden_nodes}, Output: {output_nodes}")
    
    enabled_conns = sum(1 for conn in genome.connections.values() if conn.enabled)
    print(f"Enabled connections: {enabled_conns}/{len(genome.connections)}")
    print(f"Fitness: {genome.fitness:.2f}")

def main():
    print("Snake AI Testing")
    print("1. Test NEAT model")
    print("2. Test RL model") 
    print("3. Test final model")
    
    choice = input("Choose model to test (1-3): ")
    
    if choice == "1":
        filename = "best_snake_final.pkl"
    elif choice == "2":
        filename = "rl_snake_final.pkl"
    elif choice == "3":
        filename = "snake_ai_final.pkl"
    else:
        print("Invalid choice")
        return
    
    genome = load_genome(filename)
    if not genome:
        return
    
    analyze_genome(genome)
    
    num_games = int(input("Number of test games (default 5): ") or 5)
    speed = int(input("Game speed (10-100, default 50): ") or 50)
    show_game = input("Show game visualization? (y/n, default y): ").strip().lower() != 'n'
    
    test_ai(genome, num_games=num_games, speed=speed, render=show_game)

if __name__ == "__main__":
    main()