import random
import copy
import time
import os
import sys
import tetris_base as tetris
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Constants for the genetic algorithm
POPULATION_SIZE = 20  # Number of chromosomes
NUM_GENERATIONS = 15  # Number of evolution cycles
MUTATION_RATE = 0.2   # Probability of mutation
MUTATION_RANGE = 1.0  # Range of mutation
TOURNAMENT_SIZE = 3   # Number of chromosomes in tournament selection
ELITISM_COUNT = 2     # Number of elite chromosomes to keep
NUM_ITERATIONS = 450  # Number of iterations to run each game

# Weight bounds for initialization
MIN_WEIGHT = -10.0
MAX_WEIGHT = 10.0

# Constants for the game
SEED = 42  # Random seed for reproducibility
MAX_SCORE = 999999  # Maximum score to prevent overflow

class Chromosome:
    def __init__(self, weights=None):
        """Initialize a chromosome with weights or generate random weights"""
        if weights is None:
            # Initialize with random weights
            # 1. Maximum height
            # 2. Number of holes
            # 3. Number of blocks above holes
            # 4. Number of complete lines
            # 5. Average height
            # 6. Height differences
            # 7. Well count
            # 8. Piece-to-sides contact
            # 9. Piece-to-floor contact
            # 10. Piece-to-wall contact
            self.weights = [random.uniform(MIN_WEIGHT, MAX_WEIGHT) for _ in range(10)]
        else:
            self.weights = weights
        self.fitness = 0

    def __str__(self):
        """String representation of chromosome"""
        return f"Weights: {self.weights}, Fitness: {self.fitness}"

def get_population(size):
    """Initialize a population of chromosomes"""
    return [Chromosome() for _ in range(size)]

def calculate_aggregate_height(board):
    """Calculate the sum of heights of each column"""
    total_height = 0
    for x in range(tetris.BOARDWIDTH):
        for y in range(tetris.BOARDHEIGHT):
            if board[x][y] != tetris.BLANK:
                total_height += tetris.BOARDHEIGHT - y
                break
    return total_height

def calculate_complete_lines(board):
    """Count number of complete lines"""
    complete_lines = 0
    for y in range(tetris.BOARDHEIGHT):
        if all(board[x][y] != tetris.BLANK for x in range(tetris.BOARDWIDTH)):
            complete_lines += 1
    return complete_lines

def calculate_holes(board):
    """Count number of holes (empty cells with a filled cell above them)"""
    holes = 0
    for x in range(tetris.BOARDWIDTH):
        found_block = False
        for y in range(tetris.BOARDHEIGHT):
            if board[x][y] != tetris.BLANK:
                found_block = True
            elif found_block:
                holes += 1
    return holes

def calculate_bumpiness(board):
    """Calculate the sum of differences between adjacent columns"""
    heights = []
    for x in range(tetris.BOARDWIDTH):
        height = 0
        for y in range(tetris.BOARDHEIGHT):
            if board[x][y] != tetris.BLANK:
                height = tetris.BOARDHEIGHT - y
                break
        heights.append(height)

    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness

def calculate_wells(board):
    """Calculate the sum of well depths"""
    heights = []
    for x in range(tetris.BOARDWIDTH):
        height = 0
        for y in range(tetris.BOARDHEIGHT):
            if board[x][y] != tetris.BLANK:
                height = tetris.BOARDHEIGHT - y
                break
        heights.append(height)

    wells = 0
    for i in range(len(heights)):
        if i == 0:  # Leftmost column
            wells += max(0, heights[i + 1] - heights[i])
        elif i == len(heights) - 1:  # Rightmost column
            wells += max(0, heights[i - 1] - heights[i])
        else:  # Middle columns
            wells += max(0, min(heights[i - 1], heights[i + 1]) - heights[i])
    return wells

def evaluate_board(board, weights):
    """Evaluate a board state using the weights"""
    # Calculate features
    agg_height = calculate_aggregate_height(board)
    complete_lines = calculate_complete_lines(board)
    holes = calculate_holes(board)
    bumpiness = calculate_bumpiness(board)
    wells = calculate_wells(board)

    # Apply weights to features
    score = (weights[0] * agg_height +
             weights[1] * complete_lines +
             weights[2] * holes +
             weights[3] * bumpiness +
             weights[4] * wells)

    return score

def find_best_move(board, piece, weights):
    """Find the best move for the current piece"""
    best_score = float('-inf')
    best_rotation = 0
    best_position = 0
    found_valid_move = False

    # Try each rotation
    for rotation in range(len(tetris.PIECES[piece['shape']])):
        rotated_shape = tetris.PIECES[piece['shape']][rotation]
        left_offset = 0

        for x in range(tetris.TEMPLATEWIDTH):
            if any(rotated_shape[y][x] != tetris.BLANK for y in range(tetris.TEMPLATEHEIGHT)):
                left_offset = x
                break

        for position in range(tetris.BOARDWIDTH):
            test_piece = copy.deepcopy(piece)
            test_piece['rotation'] = rotation
            test_piece['x'] = position - left_offset
            test_piece['y'] = 0

            if not tetris.is_valid_position(board, test_piece):
                continue

            while tetris.is_valid_position(board, test_piece, adj_Y=1):
                test_piece['y'] += 1

            test_board = copy.deepcopy(board)
            tetris.add_to_board(test_board, test_piece)
            lines_removed = tetris.remove_complete_lines(test_board)
            score = evaluate_board(test_board, weights)
            score += lines_removed * 100  # Bonus for lines

            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_position = position - left_offset
                found_valid_move = True

    if not found_valid_move:
        for rotation in range(len(tetris.PIECES[piece['shape']])):
            rotated_shape = tetris.PIECES[piece['shape']][rotation]
            left_offset = 0

            for x in range(tetris.TEMPLATEWIDTH):
                if any(rotated_shape[y][x] != tetris.BLANK for y in range(tetris.TEMPLATEHEIGHT)):
                    left_offset = x
                    break

            for position in range(tetris.BOARDWIDTH):
                test_piece = copy.deepcopy(piece)
                test_piece['rotation'] = rotation
                test_piece['x'] = position - left_offset
                test_piece['y'] = 0

                if tetris.is_valid_position(board, test_piece):
                    return rotation, position - left_offset

        return 0, tetris.BOARDWIDTH // 2

    return best_rotation, best_position

def play_game(weights, max_pieces=300):
    """Play a game using the provided weights"""
    board = tetris.get_blank_board()
    score = 0
    pieces_played = 0
    lines_cleared = 0

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

        while tetris.is_valid_position(board, falling_piece, adj_Y=1):
            falling_piece['y'] += 1

        tetris.add_to_board(board, falling_piece)
        lines = tetris.remove_complete_lines(board)
        lines_cleared += lines
        
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

    return score

def evaluate_fitness(population):
    """Evaluate fitness of each chromosome in the population"""
    for chromosome in population:
        chromosome.fitness = play_game(chromosome.weights)
    return population

def tournament_selection(population):
    """Select a chromosome using tournament selection"""
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament, key=lambda x: x.fitness)

def crossover(parent1, parent2):
    """Create a child by crossing over two parents"""
    # Single point crossover
    crossover_point = random.randint(1, len(parent1.weights) - 1)
    child_weights = parent1.weights[:crossover_point] + parent2.weights[crossover_point:]
    return Chromosome(child_weights)

def mutate(chromosome):
    """Mutate a chromosome"""
    for i in range(len(chromosome.weights)):
        if random.random() < MUTATION_RATE:
            chromosome.weights[i] += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
    return chromosome

def evolve(population):
    """Evolve the population to the next generation"""
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Keep track of the best chromosomes
    new_population = population[:ELITISM_COUNT]
    
    # Fill the rest of the population with children
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        
        # Create child
        child = crossover(parent1, parent2)
        child = mutate(child)
        
        new_population.append(child)
    
    return new_population

def load_best_weights_and_score(filename="best_weights.txt"):
    """Load the best weights and score from a file"""
    best_score = 0
    best_weights = None
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0 and lines[0].startswith("Score:"):
                try:
                    best_score = int(lines[0].split(":")[1].strip())
                    best_weights = []
                    for i in range(1, len(lines)):
                        if lines[i].strip():  # Skip empty lines
                            best_weights.append(float(lines[i].strip()))
                except (ValueError, IndexError):
                    # If there's an error parsing, return defaults
                    best_score = 0
                    best_weights = None
    except FileNotFoundError:
        # File doesn't exist yet, that's okay
        pass
    
    return best_score, best_weights

def save_weights_with_score(chromosome, score, filename="best_weights.txt"):
    """Save the weights and score of a chromosome to a file"""
    with open(filename, 'w') as f:
        f.write(f"Score: {score}\n")
        for weight in chromosome.weights:
            f.write(f"{weight}\n")
    print(f"Saved weights with score {score} to {filename}")

def update_best_weights(chromosome, current_run_score):
    """Update the best weights file if the new score is better"""
    # Load the existing best score and weights
    best_score, best_weights = load_best_weights_and_score()
    
    # If the new score is better, update the file
    if current_run_score > best_score:
        save_weights_with_score(chromosome, current_run_score)
        print(f"New best score: {current_run_score} (previous: {best_score})")
        return True
    else:
        print(f"Current score {current_run_score} did not exceed best score {best_score}, weights not updated")
        return False

def append_to_best_scores(generation, score, best_score, filename="best_scores.txt"):
    """Append a score entry to the best_scores.txt file"""
    with open(filename, 'a') as f:
        if generation == 0:
            f.write(f"Initial population best score: {score}\n")
        else:
            f.write(f"Generation {generation}: {score}\n")
            
        # Add a note if this is a new best score overall
        if score > best_score and generation > 0:
            f.write(f"  ** New best score! Previous: {best_score}, Improvement: +{score - best_score}\n")

def create_best_scores_file(filename="best_scores.txt"):
    """Create a new best_scores.txt file with header"""

    with open(filename, 'w') as f:
        f.write(f"===== Genetic Algorithm Run =====\n")
        f.write(f"Population size: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}\n")
        f.write(f"Mutation rate: {MUTATION_RATE}, Tournament size: {TOURNAMENT_SIZE}\n")
        f.write("=================================================\n\n")

def main():
    """Main function to run the genetic algorithm"""
    print("Starting genetic algorithm to learn Tetris...")
    
    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a fresh best_scores.txt file
    create_best_scores_file()
    
    # Load the current best score for comparison
    best_score, _ = load_best_weights_and_score()
    if best_score > 0:
        print(f"Current best score from previous runs: {best_score}")
    
    # Initialize population
    population = get_population(POPULATION_SIZE)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    population = evaluate_fitness(population)
    
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Track the best score for this run
    best_run_score = population[0].fitness
    best_run_chromosome = population[0]
    
    # Save initial best score
    append_to_best_scores(0, population[0].fitness, best_score)
    
    # Evolve for NUM_GENERATIONS
    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation+1}/{NUM_GENERATIONS}")
        print(f"Best fitness: {population[0].fitness}")
        print(f"Best weights: {population[0].weights}")
        
        # Evolve the population
        population = evolve(population)
        
        # Evaluate the new population
        print(f"Evaluating generation {generation+1}...")
        population = evaluate_fitness(population)
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best run score if needed
        if population[0].fitness > best_run_score:
            best_run_score = population[0].fitness
            best_run_chromosome = population[0]
            print(f"New best run score: {best_run_score}")
        
        # Save best score for this generation
        append_to_best_scores(generation+1, population[0].fitness, best_score)
    
    # Add a summary to the best scores file
    with open("best_scores.txt", 'a') as f:
        f.write("\n===== Final Results =====\n")
        f.write(f"Best score from this run: {best_run_score}\n")
        f.write(f"Previous best score: {best_score}\n")
        
        if best_run_score > best_score:
            f.write(f"Improvement: +{best_run_score - best_score}\n")
            f.write("Best weights file updated with new best score.\n")
        else:
            f.write(f"No improvement over previous best score (difference: {best_run_score - best_score})\n")
            f.write("Best weights file not updated.\n")
        
        f.write("=====================\n\n")
    
    # Update the best weights file if the new score is better
    updated = update_best_weights(best_run_chromosome, best_run_score)
    
    print("Genetic algorithm completed!")
    print(f"Best score: {best_run_score}")
    print(f"Best weights: {best_run_chromosome.weights}")
    print(f"Weights file updated: {updated}")

if __name__ == "__main__":
    main() 