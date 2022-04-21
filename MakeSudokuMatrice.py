import numpy as np
import cv2
from digitFinderModel import findDigit

#TODO change this algo according to segmentation output
#TODO find good threshold
def MakeSudokuMatric(listOfImages : [np.ndarray]) :
    """
    take a list of image, find the value in it and returns a sudoku as a matrice 9x9
    :param listOfImages: should be a list of np.ndarray of size [9][9]
    :return:
    """
    listOfValues = []
    # 3*255/4 means at least a quarter of the image is black
    threshold = 3 * 255 / 4
    for image in listOfImages :
        value = 0
        if isSomethingOnImage(image, threshold):
            value = findDigit(image)
        listOfValues.append(value)

    #TODO change list into sudoku
    sudoku = []
    return sudoku

def isSomethingOnImage(image : np.ndarray, theshold) -> bool :
    """
    return True if mean of image is inferior than threshold
    """
    img = image.copy()
    #if image not in grayscale -> convert it to grayscale
    if not img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if(getMeanValueOfPixel(img) <= theshold) :
        return True
    return False

def getMeanValueOfPixel(img : np.ndarray) :
    return np.mean(img)