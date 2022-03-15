import cv2

#Load face cascade

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read imported image 
img = cv2.imread('lennon.jpg')

#Convert to greyScale
grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
faces = face_cascade.detectMultiScale(grayScale)

#Draw rectangules around all faces 
for(x, y, w, z) in faces:
    cv2.rectangle(img, (x, y), (x+z , y+w), (0, 0, 255), 2)

#Display Output
cv2.imshow('img', img)
cv2.waitKey()


