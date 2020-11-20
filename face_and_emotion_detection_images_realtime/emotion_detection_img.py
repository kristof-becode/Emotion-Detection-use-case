#in this file you can detect emotions in images using deepface library

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
# ** download preatrained CNN weights ->  https://drive.google.com/uc?id=1YCox_4kJ-BYeXq27uUbasu--yz28zUMV
# ** save .h5 at 'C:\Users\Steven_Verkest/.deepface/weights/weights.h5'

img = cv2.imread('jon-snow.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img)
plt.imshow(gray)

predictions = DeepFace.analyze(img) # read **
print(predictions)
#print(predictions['dominant_emotion'])

faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades.xml') # download from github->opencv -> data-> haarcascades_frontalface_default.xml
# draw rectangle around the face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# draw text on image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,predictions['dominant_emotion'],(50,50),font,3,(0,255,0),2,cv2.LINE_4)
