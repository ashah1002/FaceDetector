import cv2
from random import randrange

#Pre-Trained Data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#IMG
#img = cv2.imread('RDJ.jpg')

webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)

while True:
    successful_frame_read, frame = webcam.read()

    #Turn RGB IMG -> Grayscale IMG
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(128,256),randrange(128,256),randrange(128,256)), 5)

    cv2.imshow('Clever Face Detector',frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()
print("Code Completed")
