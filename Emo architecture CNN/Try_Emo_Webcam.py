import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import cv2 as cv
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Keras models
# for Emopy inspired CNN trained for 50 epochs on image size (64,64)
model = load_model("/home/becode/AI/skyebase/Bagaar/model_emopy_50_ep.h5")
# # for Emopy inspired CNN trained for 10 epochs on image size (48,48); reacts slightly better to changes
#model = load_model("/home/becode/AI/skyebase/Bagaar/model_emopy_10_ep.h5")

# Emotion label for score
def emo_map(inp):
    emo = {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'disgust': 5, 'fear': 6,
           'contempt': 7}
    em = [key for key,value in emo.items() if value == inp]
    return em[0]

# Get OpenCV Cascade CLFs; eye cascade not useful here
path = '/home/becode/AI/skyebase/skyebase/lib/python3.8/site-packages/cv2/data/'
faceCascade = cv.CascadeClassifier(os.path.join(path, 'haarcascade_frontalface_default.xml'))
# eye_cascade = cv.CascadeClassifier(os.path.join(path, 'haarcascade_eye.xml'))

# Capture video feed
video_capture = cv.VideoCapture(0)
video_capture.open(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # get face
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # get eyes
    #eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2)

    # Draw a rectangle around the faces and print emotion
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]

        img_array = cv.resize(roi_gray, (64,64)) # set resize to (48,48) for 10 epoch model
        img_array = tf.expand_dims(img_array, 0)  # create batch axis
        img_array = tf.expand_dims(img_array, -1) # create input channel axis

        predictions = model.predict(img_array)
        score = predictions[0]
        sc = np.argmax(score)
        result = emo_map(sc)
        #print(result)

        cv.putText(frame, result, (x, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #0.45,(0,200,50), 2)

    # Draw a rectangle around the eyes
    #for (ex, ey, ew, eh) in eyes:
        #cv.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()