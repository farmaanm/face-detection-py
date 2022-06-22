import cv2 as cv

img = cv.imread('Images/cat.jpg')

cv.imshow('Image', img)

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_image = rescaleFrame(img)
cv.imshow('Small', resized_image)

cv.waitKey(0)