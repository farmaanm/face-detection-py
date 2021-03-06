import cv2 as cv

#reading and displaying original image
img = cv.imread('Images\group3.jpg')

#converting image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#reading xml files
haar_cascadeface = cv.CascadeClassifier('XML Files/haar_face.xml')
haar_cascadeeyes = cv.CascadeClassifier('XML Files/haar_eyes.xml')

faces_rect = haar_cascadeface.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
eyes_rect = haar_cascadeeyes.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20)

#detecting the faces and eyes
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) #lime color

for (a, b, c, d) in eyes_rect:
    cv.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), thickness=2) #blue color

#Resizing images
def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_image = rescaleFrame(img)
cv.imshow('Resized detected images', resized_image)

#Checking faces & eyes found or not
if len(faces_rect) or len(eyes_rect):
    print("Faces, eyes found")
    print(f'Number of faces found = {len(faces_rect)}')
    print(f'Number of eyes found = {len(eyes_rect)}')
else:
   print("Faces or eyes not found")

cv.waitKey(0)