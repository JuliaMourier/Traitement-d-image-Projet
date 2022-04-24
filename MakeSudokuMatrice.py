import numpy as np
import cv2
from digitFinderModel import findDigit


# TODO find good threshold
def MakeSudokuMatrice(listOfImages: [np.ndarray]):
    """
    take a list of image, find the value in it and returns a sudoku as a matrice 9x9
    :param listOfImages: should be a list of np.ndarray of size [9][9]
    :return:
    """
    listOfValues = []
    # 3*255/4 means at least a quarter of the image is black
    threshold = 240
    # threshold = 3 * 255 / 4
    for image in reversed(listOfImages):
        cv2.imshow("img", image)
        cv2.waitKey(0)
        value = 0
        if isSomethingOnImage(image, threshold):
            value = findDigit(image)
        #print(value)
        listOfValues.append(value)

    sudoku = listOfValues
    sudoku = turnListIntoSudoku(sudoku)
    return sudoku


def turnListIntoSudoku(list: []):
    i = 0
    sudoku = []
    row = []
    for elt in list:
        row.append(elt)
        i = i + 1
        if i % 9 == 0:
            sudoku.append(row)
            row = []

    return sudoku


def isSomethingOnImage(image: np.ndarray, theshold) -> bool:
    """
    return True if mean of image is inferior than threshold
    """
    img = image.copy()
    # if image not in grayscale -> convert it to grayscale
    # if not img.shape[2] == 1:
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if (getMeanValueOfPixel(img) <= theshold):
        return True
    return False


def getMeanValueOfPixel(img: np.ndarray):
    return np.mean(img)
