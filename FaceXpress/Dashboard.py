"""
This py file can be run standalone as a Streamlit app, it contains all layout and processing.
All requirements in imports below.
Keras model path should be adjusted but then also scoring part of the code-I trained on 8 emotions, neutral included.
Path to Haar cascades need to be adjusted as well
Possible additions: audio play, speech analysis, NLP, passive/active words, sentiment score..and more..
The 'NoneType' atrribute error that shows doesn't impact the app, it would be nice to get it out-related to tempfile part
Layout needs to be improved as well as some debugging in the scores.
"""


import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import os
import tempfile


# Load Keras model; change file path at wish and adjust image size accordingly down below
model = load_model("/home/becode/AI/skyebase/Bagaar/model_emopy_10_ep.h5")

# Streamlit layout
st.set_page_config(layout="wide")

# Function to map model scores to a string
def emo_map(inp):
    emo = {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'disgust': 5, 'fear': 6,
               'contempt': 7}
    em = [key for key, value in emo.items() if value == inp]
    return em[0]


# Main contains all subpages of app
def main():
    # Streamlit layout of sidebar app navigation
    st.sidebar.title("FaceXpress")

    face = cv.imread('facial.jpg',-1)
    st.sidebar.image(face)

    #st.sidebar.title("Models to select from") # kinda busy

    page = st.sidebar.selectbox(
        "Choose a model", ["Face and speech", "Seppe's math experiments"]
    )

    st.sidebar.title("About")
    st.sidebar.info(
        "This app was created for a use case with the objective "
        "of capturing micro-expressions in video feed."
    )

    # Point to functions below that stand for different app pages
    if page == "Face and speech":
        FaceSpeech()
    #elif page == "":
    else:
        Seppe()

# Facial emotion detection, use best Keras model, add speech analysis if possible
def FaceSpeech():
    ## Streamlit layout
    st.title('FaceXpress: Face and speech')
    minitext = st.text("From video feed emotions are captured and an overall emotion distribution is plotted live.")

    # File uploader
    f = st.file_uploader("Upload a video file")
    # Keep as tempfile
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    # Feed tempfile to opencv
    cap = cv.VideoCapture(tfile.name)
    cap.open(tfile.name)

    # set display for video rendering
    display = st.empty()

    # Get OpenCV Cascade CLFs; eye cascade not useful here; adjust file paths-place in same folder
    path = '/home/becode/AI/skyebase/skyebase/lib/python3.8/site-packages/cv2/data/'
    faceCascade = cv.CascadeClassifier(os.path.join(path, 'haarcascade_frontalface_default.xml'))
    # eye_cascade = cv.CascadeClassifier(os.path.join(path, 'haarcascade_eye.xml'))

    # Dictionary and keys/values lists to keep score of emotion counts
    emo_scores = {'neutral': 0, 'happiness': 0, 'surprise': 0, 'sadness': 0, 'anger': 0, 'disgust': 0, 'fear': 0,
          'contempt': 0}
    ks = [key for key in emo_scores.keys()]
    vs = [value for value in emo_scores.values()]

    # Define streamlit chart element using Plotly
    fig = go.Figure(data=[go.Bar(x=ks, y=vs)],
                   layout=go.Layout(title=go.layout.Title(text="Live emotion distribution")))
    emo_dist = st.plotly_chart(fig, use_container_width=True)

    # Loop while feed runs
    c = 1
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert to grayscale

        # For better rendering I take twice less frames
        if c % 2 == 0:
            # get face
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces and print emotion
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi_gray = gray[y:y + h, x:x + w]

                # Get prediction from grayscale face
                img_array = cv.resize(roi_gray, (48, 48))  # set resize to (48,48) for 10 epoch model
                img_array = tf.expand_dims(img_array, 0)  # create batch axis
                img_array = tf.expand_dims(img_array, -1)  # create input channel axis

                predictions = model.predict(img_array)
                score = predictions[0]
                sc = np.argmax(score)
                result = emo_map(sc)

                emo_scores[result] += 1

                vs = [value for value in emo_scores.values()]
                # Get percentages for different emotions
                perc = list(
                    map(lambda score: score * 2 / c * 100, vs))  # *2, because this part gets only accessed if c%2==True

                # Update streamlit chart element
                fig = go.Figure(
                    data=[go.Bar(x=ks, y=perc)],
                    layout=go.Layout(yaxis=dict(range=[0, 100]),
                                     title=go.layout.Title(text="Live Emotion distribution ")
                                     )
                )
                emo_dist.plotly_chart(fig, use_container_width=True)

                # Plot emotion near rectangle
                cv.putText(frame, result, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
                           2)  # 0.45,(0,200,50), 2)

            # Display the resulting frame in streamlit element
            display.image(frame, channels="BGR", use_container_width=True)

            # if cv.waitKey(1) & 0xFF == ord('q'): # Streamlit doesn't take this anymore
            # break
        c += 1

# Seppe's Math experiments
def Seppe():
    st.title("FaceXpress: Seppe's math experiments")
    # Add content


main()