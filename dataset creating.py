import cv2
import os

file = "haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(file)
folder = "data"
supfolder = input(str("dataset name : "))
sample = 1

path = os.path.join(folder,supfolder)
if not os.path.isdir(path):
    os.mkdir(path)
    (width,height)= (130,100)

cam = cv2.VideoCapture(0)
count = 1

while count < 32 :
    print(count)
    _,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130,100))
        cv2.imwrite("%s/%s.png" % (path,count),face_resize)
        count += 1

    cv2.imshow("data sets on preparing....",img)
    key = cv2.waitKey(10)
    if key ==27:
        break

cam.release
cv2.destroyAllWindows