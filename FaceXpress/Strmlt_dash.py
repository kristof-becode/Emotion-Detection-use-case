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
import plotly.graph_objects as go
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import tempfile

# Define Haar Cascades
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Streamlit layout
st.set_page_config(layout="wide")

# Function to map model scores
def prediction_map(model, k_size, gray):
    # Get prediction from grayscale face
    img_array = cv.resize(gray, k_size)  # set resize to (48,48) for 10 epoch model
    img_array = tf.expand_dims(img_array, 0)  # create batch axis
    img_array = tf.expand_dims(img_array, -1)  # create input channel axis
    predictions = model.predict(img_array)
    score = predictions[0]
    sc = np.argmax(score)
    emo = {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'disgust': 5, 'fear': 6,
           'contempt': 7}
    em = [key for key, value in emo.items() if value == sc]
    return em[0]

# Main contains all subpages of app
def main():
    # Streamlit layout of sidebar app navigation
    st.sidebar.title("FaceXpress")

    face = cv.imread('facial.jpg',-1)
    st.sidebar.image(face)

    page = st.sidebar.selectbox(
        "Choose media for emotion detection", ["Video", "Image"]
    )

    st.sidebar.title("About")
    st.sidebar.info(
        "This app was created with the objective "
        "of capturing emotions in images and video feed with a Keras CNN. "
        "The model is based on EmoPy architecture and trained on FER+ data."
    )

    # Point to functions below that stand for different app pages
    if page == "Video":
        FaceSpeech()
    #elif page == "Try an image":
     #   ImageTest()
    else:
        ImageTest()

# Facial emotion detection, use best Keras model, add speech analysis if possible
def FaceSpeech():
    ## Streamlit layout
    st.title('FaceXpress: Video emotion detection')
    minitext = st.text("From video feed emotions are captured and an overall emotion distribution is plotted live.")

    # File uploader
    f = st.file_uploader("Upload a video file")
    # Keep as tempfile
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    # Feed tempfile to opencv
    cap = cv.VideoCapture(tfile.name)
    cap.open(tfile.name)

    # Dictionary and keys/values lists to keep score of emotion counts
    emo_scores = {'neutral': 0, 'happiness': 0, 'surprise': 0, 'sadness': 0, 'anger': 0, 'disgust': 0, 'fear': 0,
          'contempt': 0}
    ks = [key for key in emo_scores.keys()]
    vs = [value for value in emo_scores.values()]

    kerasmodel_select = st.selectbox('Choose model', ['EmoPy 10 epochs', 'Emopy 50 epochs'], index=0)
    kan_button = st.button('Analyse video')
    # Load model based in selectbox
    if kerasmodel_select == 'EmoPy 10 epochs':
        model = load_model("model_emopy_10_ep.h5")
        k_size = (48, 48)
    else:
        model = load_model("model_emopy_50_ep.h5")
        k_size = (64, 64)

    # set display for video rendering
    display = st.empty()
    # Define streamlit chart element using Plotly
    fig = go.Figure(data=[go.Bar(x=ks, y=vs)],
                   layout=go.Layout(title=go.layout.Title(text="Live emotion distribution")))
    emo_dist = st.empty()#st.plotly_chart(fig)#, use_container_width=True)

    # If file uploaded and button pressed, analyse video feed
    if f is not None and kan_button:
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
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv.CASCADE_SCALE_IMAGE
                )

                # Draw a rectangle around the faces and print predicted emotion
                for (x, y, w, h) in faces:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    roi_gray = gray[y:y + h, x:x + w]
                    result = prediction_map(model,k_size, roi_gray) #emo_map(sc)
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

            c += 1

# For images
def ImageTest():

    st.title("FaceXpress: Image emotion detection")
    st.text("See what emotion is detected in an image you upload!")
    up_file = st.file_uploader("Upload an image")#,type="jpg","png")
    model_select, scale_select = st.beta_columns(2)
    model_select = model_select.selectbox('Choose model', ['EmoPy 10 epochs', 'Emopy 50 epochs'], index=0)
    scale_select = scale_select.selectbox('Choose cascade scaleFactor', [1,1.1,1.2], index=1)
    an_button = st.button('Analyse image')

    # Load model based in selectbox
    if model_select == 'EmoPy 10 epochs':
        model = load_model("/home/becode/AI/skyebase/Bagaar/model_emopy_10_ep.h5")
        size = (48, 48)
    else:
        model = load_model("/home/becode/AI/skyebase/Bagaar/model_emopy_50_ep.h5")
        size = (64, 64)
    # If file and button pressed analyse image
    if up_file is not None and an_button:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        cv_image = cv.imdecode(file_bytes, 1)
        st.image(cv_image, channels="BGR")

        # Convert to grayscale do detect faces with HAAR cascades
        gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=scale_select,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces and print predicted emotion
        for (x, y, w, h) in faces:
            cv.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]

            result = prediction_map(model,size,roi_gray)
            cv.putText(cv_image, result, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display frame in streamlit
        st.image(cv_image, channels='BGR')


main()