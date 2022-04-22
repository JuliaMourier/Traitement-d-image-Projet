from Segmentation import lecture_img, getSegmentedSudoku
from MakeSudokuMatrice import MakeSudokuMatrice
import cv2

def main():
    image = lecture_img("images/raw/sudoku.png")
    cv2.imshow("img", image)
    cv2.waitKey(0)
    segmentedSudoku = getSegmentedSudoku(image)
    print(len(segmentedSudoku))
    sudoku = MakeSudokuMatrice(segmentedSudoku)
    print(sudoku)

if __name__ == "__main__" :
    main()