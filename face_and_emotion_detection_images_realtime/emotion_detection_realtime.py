#detect your emotions via webcam realtime using deepface library

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
#download preatrained CNN weights ->  https://drive.google.com/uc?id=1YCox_4kJ-BYeXq27uUbasu--yz28zUMV
# save .h5 at 'C:\Users\Steven_Verkest/.deepface/weights/age_model_weights.h5'

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() #read one image from a video (1 frame)
    result = DeepFace.analyze(frame,actions = ['emotion'])

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    # draw rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 3, (0, 255, 0), 2, cv2.LINE_4)
    cv2.imshow('original video',frame)

    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()