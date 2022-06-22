import cv2 as cv
import numpy as np

img = cv.imread('Images/cat.jpg')
cv.imshow('cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray cat', gray)

canny = cv.Canny(img, 125, 175)
cv.imshow('canny cat', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

cv.waitKey(0)