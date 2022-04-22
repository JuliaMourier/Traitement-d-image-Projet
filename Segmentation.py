import operator

import cv2
from matplotlib import pyplot as plt
import numpy as np


def lecture_img(path):
    #image_url = "images/raw/sudoku.png"
    #img = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return img


def traitement_img(img):  # On applique ce traitement pour isoler chaque "case"
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    proc = cv2.bitwise_not(proc, proc)

    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)

    return proc


def segmentation(img):
    # trouver les contours
    _, binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)
    plt.imshow(binary, cmap="gray")
    plt.show()
    ext_contours, hierarchie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours" + str(len(ext_contours)))
    image = cv2.drawContours(img, ext_contours, -1, (0, 255, 0), 5)
    cv2.imshow('img', img)
    cv2.imshow('contours', image)

    # cv2.drawContours(img, ext_contours, -1, (0,255,0), 3)
    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            # Here we are looking for the largest 4 sided contour
            break

    # On détermine quel point correspond à quel angle
    ordered_corners = []
    bas_gauche = max(enumerate([pt[0][0] + pt[0][1] for pt in
                                ext_contours[0]]), key=operator.itemgetter(1))
    haut_gauche = min(enumerate([pt[0][0] + pt[0][1] for pt in
                                 ext_contours[0]]), key=operator.itemgetter(1))
    haut_droit = min(enumerate([pt[0][0] - pt[0][1] for pt in
                                ext_contours[0]]), key=operator.itemgetter(1))
    bas_droit = max(enumerate([pt[0][0] - pt[0][1] for pt in
                               ext_contours[0]]), key=operator.itemgetter(1))
    # l'ordre des angles est importants -- PEUT ETRE A CHANGER
    ordered_corners.append(bas_droit)
    ordered_corners.append(bas_gauche)
    ordered_corners.append(haut_gauche)
    ordered_corners.append(haut_droit)

    # on détermine les dimensions du sudoku
    Largeur_A = np.sqrt(((bas_droit[0] - bas_gauche[0]) ** 2) + ((bas_droit[1] - bas_gauche[1]) ** 2))
    Largeur_B = np.sqrt(((haut_droit[0] - haut_gauche[0]) ** 2) + ((haut_droit[1] - haut_gauche[1]) ** 2))
    width = max(int(Largeur_A), int(Largeur_B))

    hauteur_A = np.sqrt(((haut_droit[0] - bas_droit[0]) ** 2) + ((haut_droit[1] - bas_droit[1]) ** 2))
    hauteur_B = np.sqrt(((haut_gauche[0] - bas_gauche[0]) ** 2) + ((haut_gauche[1] - bas_gauche[1]) ** 2))
    height = max(int(hauteur_A), int(hauteur_B))

    # Creation de la grille du sudoku
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")
    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    perspective = cv2.warpPerspective(img, grid, (width, height))

    # here grid is the cropped image
    # grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY) # VERY IMPORTANT
    # Adaptive thresholding the cropped grid and inverting it
    # grid = cv2.bitwise_not(cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))

    # tuto : https://becominghuman.ai/sudoku-and-cell-extraction-sudokuai-opencv-38b603066066


def getSegmentedSudoku(img: np.ndarray):
    # get contours
    binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)[1]
    ext_contours, hierarchie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentedSudoku = []
    for contour in ext_contours:
        max_x = contour[0][0][0]
        min_x = contour[0][0][0]
        max_y = contour[0][0][1]
        min_y = contour[0][0][1]
        for point in contour:
            if point[0][0] > max_x:
                max_x = point[0][0]
            if point[0][0] < min_x:
                min_x = point[0][0]
            if point[0][1] > max_y:
                max_y = point[0][1]
            if point[0][1] < min_y:
                min_y = point[0][1]

        pt1 = [min_x, min_y]
        pt2 = [min_x, max_y]
        pt3 = [max_x, max_y]
        pt4 = [max_x, min_y]

        new_size_x = max_x - min_x
        new_size_y = max_y - min_y

        inputs_pts = np.float32([pt1, pt2, pt3, pt4])
        outputs_pts = np.float32([[0, 0], [0, new_size_y], [new_size_x, new_size_y], [new_size_x, 0]])

        M = cv2.getPerspectiveTransform(inputs_pts, outputs_pts)
        segment = cv2.warpPerspective(img, M, (new_size_x, new_size_y))
        #print(inputs_pts)
        #cv2.imshow("segment", segment)
        #cv2.waitKey(0)
        segmentedSudoku.append(segment)

    return segmentedSudoku

"""
img = lecture_img()
cv2.imshow('img', img)
# img = traitement_img(img)
# cv2.imshow('img', img)
cv2.waitKey(0)
# segmentation(img)
segmentedSudoku = getSegmentedSudoku(img)
for segment in segmentedSudoku:
    cv2.imshow("oui", segment)
    cv2.waitKey(0)
# cv2.waitKey(0)
"""
