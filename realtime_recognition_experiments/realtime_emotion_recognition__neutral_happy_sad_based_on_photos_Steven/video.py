# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
# from pygame import mixer
import numpy as np
import imutils
import time
import cv2
import os
import math
from PIL import ImageGrab

# system libraries
import os
import sys
from threading import Timer
import shutil
import time
import pandas as pd


# dataframe to safe timestamps, and emotions in at that time, so you can plot over time the emotions
timestamps = []
emotions = []

df = pd.DataFrame()
print(df)


# detect surprise/disgust/anger

detections = None


def detect_and_predict_mask(frame, faceNet, maskNet, threshold):
    # grab the dimensions of the frame and then construct a blob
    # from it
    global detections
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            locs.append((startX, startY, endX, endY))
            # print(maskNet.predict(face)[0].tolist())
            preds.append(maskNet.predict(face)[0].tolist())
    return (locs, preds)


# SETTINGS
MASK_MODEL_PATH = os.getcwd() + "\\model\\model_realtime_emotion_recognition__neutral_happy_sad_based_on_photos_Steven.h5"
FACE_MODEL_PATH = os.getcwd() + "\\face_detector"
SOUND_PATH = os.getcwd() + "\\sounds\\alarm.wav"
THRESHOLD = 0.5

# Load Sounds
# mixer.init()
# sound = mixer.Sound(SOUND_PATH)

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
weightsPath = os.path.sep.join([FACE_MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading emotion detector model...")
maskNet = load_model(MASK_MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture('whatmakesyouunique_robin.mp4')  # vs = VideoStream(0).start()

# make object that can save frames into video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # https://cloudconvert.com/webm-to-mp4 -> convert .webm to .mp4 on this site
out = cv2.VideoWriter('unique_robin.mp4', fourcc, 20.0, (640, 480))
time.sleep(2.0)

labels = ["happy", "neutral", "sad"]

# loop over the frames from the video stream
while True:
    start = time.time()
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cap.read()  # frame = vs.read()
    frame = imutils.resize(frame, width=400)
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, THRESHOLD)

    # capture screen , to save it later
    img = ImageGrab.grab()
    img_np = np.array(img)
    frame_ = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame_)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        # include the probability in the label
        label = str(labels[np.argmax(pred)])
        # display the label and bounding box rectangle on the output

        # save screeshots into mp4
        out.write(frame_)
        cv2.imshow('output', frame)  # show what is going on

        stop = time.time()
        timestamp = stop - start

        # frame
        if label == "anger":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 50), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 200, 50), 2)
            emotions.append('anger')
            timestamps.append(timestamp)
        elif label == "disgust":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                        2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            emotions.append('disgust')
            timestamps.append(timestamp)
        elif label == "fear":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 50, 200), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 50, 200), 2)
            emotions.append('fear')
            timestamps.append(timestamp)
        elif label == "happy":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                        2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            emotions.append('happy')
            timestamps.append(timestamp)
        elif label == "neutral":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 50, 200), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 50, 200), 2)
            emotions.append('neutral')
            timestamps.append(timestamp)
        elif label == "sad":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                        2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            emotions.append('sad')
            timestamps.append(timestamp)
        elif label == "surprise":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 50, 200), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 50, 200), 2)
            emotions.append('surprise')
            timestamps.append(timestamp)

    # show the output frame
    frame = cv2.resize(original_frame, (860, 490))
    cv2.imshow("Facial Expression", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break



df['timestamp'] = timestamps
df['emotion'] = emotions

print(df.head())
# saving the dataframe
df.to_csv('time_emotion.csv')


# do a bit of cleanup
cv2.destroyAllWindows()
# vs.stop()