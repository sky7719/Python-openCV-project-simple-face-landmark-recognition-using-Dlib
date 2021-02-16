# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:53:08 2021

@author: shivam kumar
"""

import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture()

address = "http://192.168.1.4:8080/video"
cap.open(address)

while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        y1 = face.top()
        x1 = face.left()
        y2 = face.bottom()
        x2 = face.right()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(130,49,109),1)
        landmark = predictor(gray,face)
        for i in range(0,67):
            x = landmark.part(i).x
            y = landmark.part(i).y

            cv2.circle(frame,(x,y),1,(0,200,0),-1)
        
    cv2.imshow("image",frame)
    if cv2.waitKey(200)==13:
        break
cap.release()
cv2.destroyAllWindows()