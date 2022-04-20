import math

import numpy as np


# Main part, called by main
def backtracking_csp_algorithm(sudoku: []):
    # Display the depth of the algorithm, (9 ^ number of empty squares)
    nb_empty_squares = number_of_empty_squares(sudoku)
    print("Depth : " + str(nb_empty_squares) + " empty squares => 9^" + str(nb_empty_squares) + " = " + str(
        math.pow(9, nb_empty_squares)))
    return solve(sudoku)


# Recursive function which solve the sudoku if it is soluble
def solve(sudoku: []):
    # if there is no empty square :
    if is_sudoku_filled(sudoku):
        # Return if the sudoku is soluble and the solution if exists
        return is_sudoku_valid(sudoku), sudoku
    else:  # if the sudoku is not filled
        # Get the indexes of the most constrained empty square (MRV)
        i_choice, j_choice = get_most_constrained_square(sudoku)
        # Try every possibility (between 1 and 9) for this square
        for i in range(1, len(sudoku) + 1):
            sudoku[i_choice][j_choice] = i
            # If the solution is valid, continue; else pass to next numbers (allowing to test only the possible grids)
            if is_sudoku_valid(sudoku):
                # Continue to solve the new sudoku
                is_solved, sudoku = solve(sudoku)
                if is_solved:
                    # Return that the sudoku is soluble and the solution
                    return True, sudoku
        # else (if any number fit)
        # Remove the last modification
        sudoku[i_choice][j_choice] = 0
        # Return that the sudoku is insoluble
        return False, sudoku


# Test if the sudoku is filled, if at least a square is empty return False, return True otherwise
def is_sudoku_filled(sudoku: []):
    if np.count_nonzero(sudoku == 0):
        return False
    return True


# Return the number of empty squares
def number_of_empty_squares(sudoku: []):
    counter = np.count_nonzero(sudoku == 0)
    return counter


# Return the indexes of the first empty square
def get_indexes_of_next_empty_square(sudoku: []):
    for i in range(len(sudoku)):
        for j in range(len(sudoku)):
            if sudoku[i][j] == 0:
                return i, j
    return True


# Test if the sudoku is valid : test every rows, columns and blocks
def is_sudoku_valid(sudoku: []):
    for i in range(len(sudoku)):
        if not is_part_valid(sudoku[i]):
            return False
        if not is_part_valid(get_sudoku_block(sudoku, i)):
            return False
        if not is_part_valid(get_sudoku_column(sudoku, i)):
            return False

    return True


# Return the numbers on column k
def get_sudoku_column(sudoku: [], k: int):
    column = []
    for i in range(len(sudoku)):
        for j in range(len(sudoku)):
            if j == k:
                column.append(sudoku[i][j])
    return column


# Test if a part is valid : if there is only 1 iteration of a number in the part
def is_part_valid(row: []):
    existing_numbers = []
    for n in row:
        if n != 0:  # only if the square is not empty
            # If the table contains already the number, returns false
            if existing_numbers.__contains__(n):
                return False
            else:  # Add the number to the table otherwise
                existing_numbers.append(n)
    return True


# Get the number in the block k (0 to 8)
def get_sudoku_block(sudoku: [], k: int):
    block = []
    # Math trick (trust me it works)
    for i in range((k % 3) * 3, (k % 3) * 3 + 3):
        for j in range(3 * math.floor(k / 3), 3 * math.floor(k / 3) + 3):
            block.append(sudoku[i][j])
    return block


# Return an empty 9x9 matrix
def get_empty_matrix_9x9():
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])


# Calculate the number of non empty square in a part
def calculate_non_empty_number_of_squares(part: []):
    # Initialisation
    count = 0
    for i in range(len(part)):
        # if the square is not empty, increment count
        if part[i] != 0:
            count += 1
    return count


# Calculate the constraint matrix
def calculate_constraint_matrix(sudoku: []):
    # Initialisation
    constraint_matrix = get_empty_matrix_9x9().copy()
    rainbow_table_column = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    rainbow_table_block = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Calculate the number of non empty squares for column, row
    for i in range(len(sudoku)):
        for j in range(len(sudoku)):
            # For columns :
            if sudoku[j][i] == 0:
                if rainbow_table_column[i] == 0:
                    # Save in table in order to save time
                    rainbow_table_column[i] = calculate_non_empty_number_of_squares(get_sudoku_column(sudoku, i))
                constraint_matrix[j][i] += rainbow_table_column[i]
            # For rows :
            if sudoku[i][j] == 0:
                constraint_matrix[i][j] += calculate_non_empty_number_of_squares(sudoku[i])
        # For blocks :
        for m in range((i % 3) * 3, (i % 3) * 3 + 3):
            for n in range(3 * math.floor(i / 3), 3 * math.floor(i / 3) + 3):
                if sudoku[m][n] == 0:
                    if rainbow_table_block[i] == 0:
                        # Save in table in order to save time
                        rainbow_table_block[i] = calculate_non_empty_number_of_squares(get_sudoku_block(sudoku, i))
                    constraint_matrix[m][n] += rainbow_table_block[i]
    return constraint_matrix


# Find the most constrained square in the constraint matrix and return its indexes
def get_most_constrained_square(sudoku: []):
    matrix = calculate_constraint_matrix(sudoku)
    index = np.where(matrix == np.amax(matrix))
    listofIndices = list(zip(index[0], index[1]))
    return listofIndices[0]
