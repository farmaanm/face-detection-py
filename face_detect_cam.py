import cv2 as cv

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

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) #lime color

    for (a, b, c, d) in eyes:
        cv.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), thickness=2) #blue color

    cv.imshow('Video capture', img)

    k = cv.waitKey(30) & 0xff
    if k==27:
        break

    for z in range(0,10):
    # press the letter "q" to save the picture
        if (len(faces) or len(eyes)):
            # write the captured image with this name
            img_name = "take{}.jpg".format(z)
            cv.imwrite(img_name,img)

    if (len(faces) or len(eyes)):
        print(f'{len(faces)} faces, {len(eyes)} eyes found ')
    else:
        print("Faces not found")

cap.release()