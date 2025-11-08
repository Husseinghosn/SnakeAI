import pygame
from game import SnakeGame
from AI import load_trained_model
import time

def main():
    # Create game with AI mode enabled
    game = SnakeGame(w=500, h=500, ai_mode=True)
    
    print("Starting AI Snake Game!")
    print("Press 'A' to toggle AI mode on/off")
    print("Press 'Q' to quit")
    
    # Game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            print(f"Game Over! Final Score: {score}")
            print("Restarting in 2 seconds...")
            time.sleep(2)
            
            # Reset game
            game = SnakeGame(w=500, h=500, ai_mode=True)
        
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                return

if __name__ == "__main__":
    main()