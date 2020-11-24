## Emotion Detection: a use case ..about detecting microexpressions

This is a use case collaboration of Steven, Robin, Seppe and Kristof

## Summary
This repo is the result of a collaborative work on a use case with the objective of investigating the possibility of detecting microexpressions in images and videos. Implementations of these findings could prove useful in HR interviewing of possible job candidates by providing extra information to evaluate and analyse.

Applying machine learning to this problem proved to be very challenging as there are very little (video)datasets to train a model to recognize and classify microexpressions. These datasets do exist, i.e. SMIC and CASME, but are restricted to research purposes and limited in size.

Therefore we decided to take a more traditional approach and build several Tensorflow Keras CNN models to detect emotions trained on images in the FER+ dataset. 
As these have severe limitations when used to classify emotions in images and video we decided to investigate Landmark segmentation to track microexpressions. Again, with lack of good training data it was difficult to take this approach further. 

An adverserial approach was tested to try and fool the Keras models as they mostly labeled images as sad or angry.
We also explored and played around with some API's such as Amazon Rekognition, Microsoft Azure and some popular Python libraries such as DeepFace which uses several recent pre-trained models. 

Lastly a very simple Streamlit dashboard was created to present some of our results.

## Table of contents

* [Microexpressions](##project-goal)
* [Technologies used](#technologies)
* [Approaches](#general-info)
* [Data: FER+](#general-info)
* [EmoPy architecture CNN](#general-info)
* [MobileNetV2 CNN](#general-info)
* [Facial landmark segementation](#general-info)
* [APIs](#general-info)
* [Adverserial FER](#general-info)
* [Streamlit dashboard](#general-info)
* [To use](#general-info)

## Microexpressions

A microexpression is a very brief, involuntary facial expression humans make when experiencing
an emotion. They usually last 0.5–4.0 seconds and cannot be faked.
Dr. Paul Ekman popularized the term “microexpression” and greatly expanded the research.

A few examples:

| Happy | Fear |
|-------|------|
|Corners of the lips are drawn back and up | Eyebrows are raised and drawn together, usually in a flat line |
|Mouth may or may not be parted, teeth exposed |Wrinkles in the forehead are in the center between the eyebrows, not across |
|A wrinkle runs from outer nose to outer lip |Upper eyelid is raised, but the lower lid is tense and drawn up |
|Cheeks are raised |Eyes have the upper white showing, but not the lower white  |
|Lower eyelid may show wrinkles or be tense |Mouth is open and lips are slightly tensed or stretched and drawn back |

## Technologies used
- OpenCV for image and video handling
- DeepFace
- Tensorflow Keras
- MobileNetV2
- Adverserial Networks
- Dlib for Facial landmarks
- Streamlit
- Microsoft Azure
- ...

## Approaches
- build Keras CNN for emotion recognition based on EmoPy CNN architecture
- build Keras CNN for emotion recognition with transfer learning using pretrained MobileNetV2
- experiment with facial landmark segmentation using Dlib
- trying to detect faces and emotions with deepFace
- trying Azure and Amazon Rekognition Facial Emotion Rekognition APIs
- trying Adversarial FER to protect against FER models

## Data: FER+

The FER+ annotations provide a set of new labels for the standard Emotion FER dataset. In FER+, each image has been labeled by 10 crowd-sourced taggers, which provide better quality ground truth for still image emotion than the original FER labels. The image emotion labels are: hapiness, anger, fear, disgust, contempt, surprise, sadness and neutral.

Here are some examples of the FER vs FER+ labels extracted from the abovementioned paper (FER top, FER+ bottom):

You can find all necessary information on FER+ and how to generate it in this repo: https://github.com/microsoft/FERPlus

The original FER data set can be found here: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## EmoPy architecture CNN

EmoPy is an open source project to detect emotions in images and videos. We chose to build our own Keras model based on the CNN architecture that is proposed as most performant, using batch normalisation and dropout layers.

The model was trained on FER+ images, using majority voting on the 7 emotion labels plus neutral label. Training on +- 29k images and tested/validated on +- 7k images.
The accuracy was > 70% for training and testing/validation.

<table>
  <tr>
    <td><img src="https://github.com/kristof-becode/3D-House/blob/master/OLV.png" width=75% height=75%/></td>
    <td><img src="https://github.com/kristof-becode/3D-House/blob/master/Korenmarkt%2015.png" width=75% height=75%/></td>
    <td><img src="https://github.com/kristof-becode/3D-House/blob/master/Snoekstraat.png" width=75% height=75%/></td>
  </tr>
 </table>



## MobileNetV2 CNN

A Keras model was created using the pretrained MobileNetV2 with transfer learning. The upper layers of this model was trained on FER+ with 7 emotions.

The accuracy:

<p align="center">
  <img src="https://github.com/kristof-becode/Machine-Learning/blob/master/Scikit_learn_logo.png" width=25% >
</p>


## Facial landmark segementation

Using Dlib facial landmarks were segmented. 
To evaluate facial expressions linked to different emotions, detected faces in images and video needed to be standardized in the frame. 
Changes in positioning of eye brows and lips could be detected. More video training data is needed to take this approach further.

<table>
  <tr>
    <td><img src="https://github.com/kristof-becode/3D-House/blob/master/OLV.png" width=75% height=75%/></td>
    <td><img src="https://github.com/kristof-becode/3D-House/blob/master/Korenmarkt%2015.png" width=75% height=75%/></td>
    <td><img src="https://github.com/kristof-becode/3D-House/blob/master/Snoekstraat.png" width=75% height=75%/></td>
  </tr>
 </table>

## APIs

The emotion labeling on images was compared using Amazon Rekognition and the API from Microsoft Azure.
Next to this the labeling with DeepFace on images was explored with several pretrained models such as FaceNet and VGG-Face.

## Adverserial FER

An adverserial Keras model using FGSM was implemented to investigate fooling our earlier Keras models.

<p align="center">
  <img src="https://github.com/kristof-becode/Machine-Learning/blob/master/Scikit_learn_logo.png" width=25% >
</p>

## Streamlit dashboard

A simple implementation of the Keras EmoPy inspired model in Streamlit. Images and videos can be uploaded to classify with the according emotion. OpenCV was used for image/video handling and to return a frames with emotion labeling.

<p align="center">
  <img src="https://github.com/kristof-becode/Machine-Learning/blob/master/Scikit_learn_logo.png" width=25% >
</p>


## To use

To build a Keras EmoPy inspired CNN model trained on the FER2013+ dataset use `Emo_model.py` located in the `Emo architecture CNN` folder.
```
cd EmoarchitectureCNN
python Emo_model.py
```

To build a Keras CNN FER model for real-time detection use `detect_emotion_video.py`  This will open a webcam and give FER overlaid on the webcam stream.
```
python detect_emotion_video.py
```

To detect facial landmarks, try the jupyter notebook `EmotionRecognition.ipynb` in the `FacialLandmarks` folder.
```
cd FacialLandmarks
EmotionRecognition.ipynb
```

To detect emotions from what is verbally said in a video using IBM Watson and NRCLex
```
mp4_to_emotion.py
```

To generate adversarial FER images, try the jupyter notebook `Adversarial_FER.ipynb` in the `Adversarial_FER` folder

To detect emotions in realtime
```
all the file-names that start with: realtime_emotion_recognition....

(the end of the file-name describes which dataset was used to train the model)
```

To render the Streamlit dashboard for emotion detection in images/videos:
```
cd FaceXpress
streamlit run Strmlt_dash.py
```

