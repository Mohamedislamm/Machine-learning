import random
import copy
import time
import sys
import tetris_base as tetris
import pygame
from datetime import datetime
from genetic_algorithm import (calculate_aggregate_height, calculate_complete_lines,
                             calculate_holes, calculate_bumpiness, calculate_wells,
                             evaluate_board, find_best_move)

# Constants for the game
SEED = 42  # Random seed for reproducibility
NUM_ITERATIONS = 600  # Number of iterations for the final run
MAX_SCORE = 999999  # Maximum score to prevent overflow
VISUALIZATION_SPEED = 50  # Speed of visualization (ms)

def load_weights(filename="best_weights.txt"):
    """Load the weights from a file"""
    weights = []
    best_score = 0
    
    try:
        with open(filename, 'r') as f:

            lines = f.readlines()
            
            # Check if the first line contains the score
            if lines and lines[0].strip().startswith("Score:"):
                try:
                    best_score = int(lines[0].split(":")[1].strip())
                    # Skip the first line (score) and process weights
                    for line in lines[1:]:
                        if line.strip():  # Skip empty lines
                            weights.append(float(line.strip()))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing score from weights file: {e}")
            else:
                # Old format without score, just read all lines as weights
                for line in lines:
                    if line.strip():  # Skip empty lines
                        weights.append(float(line.strip()))
    except FileNotFoundError:
        print(f"Weights file {filename} not found. Please run genetic_algorithm.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
    
    if not weights:
        print("No weights found in the weights file. Please run genetic_algorithm.py first.")
        sys.exit(1)
        
    print(f"Loaded weights with score: {best_score}")
    return weights, best_score

def play_game(weights, max_pieces=300, record_moves=False):
    """Play a game using the provided weights"""
    board = tetris.get_blank_board()
    score = 0
    pieces_played = 0
    moves_history = [] if record_moves else None

    falling_piece = tetris.get_new_piece()
    next_piece = tetris.get_new_piece()

    while pieces_played < max_pieces:
        if falling_piece is None:
            falling_piece = next_piece
            next_piece = tetris.get_new_piece()
            pieces_played += 1

            if not tetris.is_valid_position(board, falling_piece):
                break
        
        rotation, x_pos = find_best_move(board, falling_piece, weights)
        falling_piece['rotation'] = rotation
        falling_piece['x'] = x_pos

        if record_moves:
            # Record the move before executing it
            moves_history.append({
                'piece': copy.deepcopy(falling_piece),
                'next_piece': copy.deepcopy(next_piece),
                'board': copy.deepcopy(board),
                'score': score
            })

        while tetris.is_valid_position(board, falling_piece, adj_Y=1):
            falling_piece['y'] += 1

        tetris.add_to_board(board, falling_piece)
        lines = tetris.remove_complete_lines(board)
        
        # Update score using official Tetris scoring from tetris_base.py
        if lines == 1:
            score += 40
        elif lines == 2:
            score += 100
        elif lines == 3:
            score += 300
        elif lines == 4:
            score += 1200
            
        falling_piece = None

    return score, pieces_played, moves_history

def visualize_game(moves_history):
    """Visualize a recorded game"""
    pygame.init()
    pygame.display.set_caption('Tetris AI - Best Game Replay')
    tetris.FPSCLOCK = pygame.time.Clock()
    tetris.DISPLAYSURF = pygame.display.set_mode((tetris.WINDOWWIDTH, tetris.WINDOWHEIGHT))
    tetris.BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    tetris.BIGFONT = pygame.font.Font('freesansbold.ttf', 100)

    running = True
    move_index = 0
    
    try:
        while running and move_index < len(moves_history):
            move = moves_history[move_index]
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Space to pause/unpause
                        paused = True
                        while paused and running:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif pause_event.type == pygame.KEYUP:
                                    if pause_event.key == pygame.K_ESCAPE:
                                        running = False
                                        paused = False
                                    elif pause_event.key == pygame.K_SPACE:
                                        paused = False

            if running:
                # Calculate current level based on score
                try:
                    level, _ = tetris.calc_level_and_fall_freq(move['score'])
                except:
                    level = 1  # Fallback if calculation fails

                # Draw the game state
                tetris.DISPLAYSURF.fill(tetris.BGCOLOR)
                tetris.draw_board(move['board'])
                tetris.draw_status(move['score'], level)
                tetris.draw_next_piece(move['next_piece'])
                if move['piece'] is not None:
                    tetris.draw_piece(move['piece'])
                
                # Draw instructions
                instructions_surf = tetris.BASICFONT.render('SPACE to pause, ESC to exit', True, tetris.TEXTCOLOR)
                instructions_rect = instructions_surf.get_rect()
                instructions_rect.topleft = (20, 20)
                tetris.DISPLAYSURF.blit(instructions_surf, instructions_rect)
                
                pygame.display.update()
                pygame.time.delay(VISUALIZATION_SPEED)
                tetris.FPSCLOCK.tick(tetris.FPS)
                
                move_index += 1

    finally:
        pygame.quit()

def save_final_score(score, prev_best_score, iterations):
    """Save the final score to a file"""
    with open('final_score.txt', 'w') as f:
        f.write("===== Tetris Optimal Run Results =====\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Iterations completed: {iterations}/{NUM_ITERATIONS}\n\n")
        f.write(f"Final score: {score}\n")
        f.write(f"Previous best score: {prev_best_score}\n")
        f.write(f"Difference from best: {score - prev_best_score}\n\n")
        f.write(f"Game over after {iterations} iterations\n")

def main():
    """Main function to run the optimal Tetris AI"""
    print("\nLoading best weights...")
    weights, prev_best_score = load_weights()
    
    print("\nRunning game with best weights...")
    random.seed(SEED)  # Use fixed seed for reproducibility
    score, iterations, moves_history = play_game(weights, NUM_ITERATIONS, record_moves=True)
    print(f"Game completed. Score: {score}")
    
    # Save the final score
    save_final_score(score, prev_best_score, iterations)
    print("\nResults saved to final_score.txt")
    
    # Visualize the game
    print("\nControls:")
    print("- SPACE: Pause/Unpause")
    print("- ESC: Exit visualization")
    print("\nStarting game visualization...")
    visualize_game(moves_history)

if __name__ == '__main__':
    main() 