import numpy as np
import cv2


def getSudokuGridFromImage(image: np.ndarray):
    """
    function that takes an image, finds the sudoku grid and returns the area found
    """

    # copy image
    img = image.copy()

    # check if sudoku is in grayscale, if not : transform to grayscale
    if not len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # apply gaussian threshold
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # find all the contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    # select the largest contour
    max_cnt = max(contours, key=cv2.contourArea)

    # calculte the perimeter of the contour and try to appriximate vertices
    peri = cv2.arcLength(max_cnt, True)
    approxVertices = cv2.approxPolyDP(max_cnt, 0.015 * peri, True)
    # flatten the vertices array
    pts = np.squeeze(approxVertices)
    # find height and width of the grid
    grid_height = np.max(pts[:, 1]) - np.min(pts[:, 1])
    grid_width = np.max(pts[:, 0]) - np.min(pts[:, 0])

    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)

    #input corners points to warp grid
    bounding_rect = np.array([pts[np.argmin(sum_pts)],
                              pts[np.argmin(diff_pts)],
                              pts[np.argmax(sum_pts)],
                              pts[np.argmax(diff_pts)]], dtype=np.float32)
    #output corners points to warp grid
    dst = np.array([[0, 0],
                    [grid_width - 1, 0],
                    [grid_width - 1, grid_height - 1],
                    [0, grid_height - 1]], dtype=np.float32)

    # apply warp perspective to get the grid
    M = cv2.getPerspectiveTransform(bounding_rect, dst)
    bw_image = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)[1]
    warped_img = cv2.warpPerspective(bw_image, M, (grid_width, grid_height))
    blur_warped = cv2.GaussianBlur(warped_img, (5, 5), 0)

    #bw_warped_image = cv2.threshold(warped_img, 135, 255, cv2.THRESH_BINARY)[1]

    #return bw_warped_image
    return blur_warped

