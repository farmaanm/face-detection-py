import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#connecting the cascade files
haar_cascadeface = cv.CascadeClassifier('XML Files/haar_face.xml')
haar_cascadeeyes = cv.CascadeClassifier('XML Files/haar_eyes.xml')

#accessing the webcam (every webcam has a number, default is 0)
cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # to detect faces in video
    faces = haar_cascadeface.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    eyes = haar_cascadeeyes.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20)

    # setting threshold of gray image
    _, threshold = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv.findContours(
	    threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    i = 0




    #----------------------------------------------------------------------------------

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) #lime color

    for (a, b, c, d) in eyes:
        cv.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), thickness=2) #blue color


    #----------------------------------------------------------------------------------



    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv.approxPolyDP(
            contour, 0.01 * cv.arcLength(contour, True), True)
        
        # using drawContours() function
        cv.drawContours(img, [contour], 0, (0, 255, 0), 1)

    cv.imshow('Video capture', img)

    k = cv.waitKey(30) & 0xff
    if k==27:
        break

cap.release()