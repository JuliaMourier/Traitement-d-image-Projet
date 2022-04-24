import numpy as np
import cv2

from Segmentation import lecture_img, getSegmentedSudoku
from MakeSudokuMatrice import MakeSudokuMatrice
from SudokuSolver.backtracking_csp_algorithm import backtracking_csp_algorithm
from getSudokuGridFromImage import getSudokuGridFromImage

def main():
    #image = lecture_img("images/raw/sudoku.png")
    image = lecture_img("images/raw/realSudoku.jpg")
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

    canBeSolved, solvedSudoku = backtracking_csp_algorithm(npSudoku)
    print(canBeSolved)
    print(solvedSudoku)

if __name__ == "__main__" :
    main()