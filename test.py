import pygame
from game import SnakeGame
import time

def test_best_model():
    # Create game with AI mode enabled by default
    game = SnakeGame(w=500, h=500)
    game.ai_mode = True
    
    # Try to load the best model
    try:
        from AI import load_trained_model
        game.ai_model = load_trained_model()
        print("Loaded trained model successfully!")
        print(f"Model architecture: {game.ai_model}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Running with random weights...")
    
    print("Starting AI Test!")
    print("Press 'A' to toggle AI mode")
    print("Press 'Q' to quit")
    print("Press 'R' to reset game")
    
    total_games = 0
    total_score = 0
    best_score = 0
    
    # Game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            total_games += 1
            total_score += score
            if score > best_score:
                best_score = score
            
            avg_score = total_score / total_games if total_games > 0 else 0
            
            print(f"Game {total_games}: Score = {score}, Best = {best_score}, Average = {avg_score:.2f}")
            print("Restarting in 2 seconds...")
            time.sleep(2)
            
            # Reset game
            game = SnakeGame(w=500, h=500)
            game.ai_mode = True
            if 'ai_model' in locals():
                game.ai_model = ai_model
        
        # Check for quit or reset
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    return
                elif event.key == pygame.K_r:
                    print("Manual reset!")
                    game = SnakeGame(w=500, h=500)
                    game.ai_mode = True
                    if 'ai_model' in locals():
                        game.ai_model = ai_model

if __name__ == "__main__":
    test_best_model()