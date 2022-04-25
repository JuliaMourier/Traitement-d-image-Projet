import numpy as np
import cv2
import argparse

from Segmentation import lecture_img, getSegmentedSudoku
from MakeSudokuMatrice import MakeSudokuMatrice
from SudokuSolver.backtracking_csp_algorithm import backtracking_csp_algorithm
from getSudokuGridFromImage import getSudokuGridFromImage
from SudokuSolver.View import View
from checkNumber import checkSudoku


def main():
    # We first create the ArgumentParser object
    # The created object 'parser' will have the necessary information
    # to parse the command-line arguments into data types.
    parser = argparse.ArgumentParser()

    # Add 'path_image_input' argument using add_argument() including a help. The type is string (by default):
    parser.add_argument("path_image_input", help="path to input image to be displayed")

    # Parse the argument and store it in a dictionary:
    args = vars(parser.parse_args())

    INPUT_PATH = args["path_image_input"]
    # Parameter to put in Config : "images/raw/realSudoku.jpg"
    #image = lecture_img("images/raw/sudoku.png")
    image = lecture_img(INPUT_PATH)
    image = cv2.resize(image, (450,450))
    cv2.imshow("img", image)
    cv2.waitKey(0)
    grid = getSudokuGridFromImage(image)
    cv2.imshow("grid", grid)
    cv2.waitKey(0)
    segmentedSudoku = getSegmentedSudoku(grid)
    #segmentedSudoku = getSegmentedSudoku(image)
    print(len(segmentedSudoku))
    sudoku = MakeSudokuMatrice(segmentedSudoku)
    #make sudoku a numpy array
    npSudoku = np.array(sudoku)
    print("get sudoku :")
    print(npSudoku)
    print("solve sudoku :")

    npSudoku = checkSudoku(npSudoku)

    canBeSolved, solvedSudoku = backtracking_csp_algorithm(npSudoku)

    # If it cannot be solved :
    if not canBeSolved:
        print("This sudoku is insolvable")
    else:
        # Display the solution
        View(solvedSudoku)

if __name__ == "__main__" :
    main()
