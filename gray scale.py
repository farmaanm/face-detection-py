import cv2 as cv
import numpy as np

img = cv.imread('Images/group3.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray image', gray)

blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('Blur image', blur)

canny = cv.Canny(img, 125, 175)
cv.imshow('Canny image', canny)

resized = cv.resize(img, (700, 700))
cv.imshow('Resized image', resized)

cv.waitKey(0)