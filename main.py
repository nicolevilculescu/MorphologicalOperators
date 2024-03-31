# Write an application that uses morphological operators to extract corners from an image using the following algorithm:
#
# 1.       R1 = Dilate(Img,cross)
#
# 2.       R1 = Erode(R1,Diamond)
#
# 3.       R2 = Dilate(Img,Xshape)
#
# 4.       R2 = Erode(R2,square)
#
# 5.       R = absdiff(R2,R1)
#
# 6.       Display(R)
#
# Transform the input image to make it compatible with binary operators and display the results imposed over the
# original image. Apply the algorithm at least on the following images Rectangle and Building.

import cv2
import numpy as np

imgRectangle = cv2.imread("square-rectangle.png", cv2.IMREAD_GRAYSCALE)
imgBuilding = cv2.imread("MorpologicalCornerDetection.png", cv2.IMREAD_GRAYSCALE)

# edge detection
kernel = np.ones((3, 3), np.uint8)

edgesRectangle = cv2.morphologyEx(imgRectangle, cv2.MORPH_GRADIENT, kernel)
edgesBuilding = cv2.morphologyEx(imgBuilding, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('edges rectangle', edgesRectangle)
cv2.imshow('edges building', edgesBuilding)

cv2.waitKey(0)

# building the structural elements
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
xshape = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]], dtype=np.uint8)
rhombus = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)


def detecting_corners(img):
    r1 = cv2.dilate(img, cross)
    r1 = cv2.erode(r1, rhombus)
    r2 = cv2.dilate(img, xshape)
    r2 = cv2.erode(r2, square)
    dst = cv2.absdiff(r2, r1)
    ret, dst = cv2.threshold(dst, 40, 255, cv2.THRESH_BINARY)

    cv2.imshow('corners', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detecting_corners(imgRectangle)
detecting_corners(imgBuilding)
