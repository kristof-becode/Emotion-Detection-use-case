## Emotion Detection: a use case ..about detecting microexpressions

This is a use case collaboration of Steven, Robin, Seppe and Kristof

## Summary
This repo is the result of a collaborative work on a use case with the objective of investigating the possibility of detecting micro-expressions in images and videos. Implementations of these findings could prove useful in HR interviewing of possible job candidates by providing extra information to evaluate and analyse.

Applying machine learning to this problem proved to be very challenging as there are very little (video)datasets to train a model to recognize and classify micro-expressions. These datasets do exist, i.e. SMIC and CASME, but are restricted to research purposes and limited in size.
Therefore we decided to take a more traditional approach and build several Keras CNN models to detect emotions trained on images in the Fer+ dataset. 
As these have severe limitations when used to classify emotions in images and video we decided to investigate Landmark segmentation to track micro-expressions. Again, with lack of good training data it was difficult to take this approach further. 
An adverserial approach was tested to try and fool the Keras models as they mostly labeled images as sad or angry.
We also explored and played around with some API's such as Amazon Rekognition, Microsoft Azure and some popular Python libraries such as DeepFace which uses several recent pre-trained models. A very simple Streamlit dashboard was created to present some of our results.



## Table of contents
* [Project Goal](#general-info)
* [Approach](#general-info)
* [Background information](#general-info)
* [Microexpression examples](#general-info)
* [Results](#results)
* [Technologies](#technologies)
* [Content](#content)

## Project Goal
We try to detect facial microexpressions.

## Approach
- trying to detect faces (with deepFace)
- trying to detect emotions in faces, training own CNN & MobilNetV2
- trying to train a model using face landmarks
- trying Azure and Amazon Rekognition Facial Emotion Rekognition APIs
- trying Adversarial FER to protect against FER models

## Background information

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

## Results
- was able to detect neutral , happy, emotions trained on Stevens face in realtime using the webcam, model had accuracy of 75% *
- trained the MobileNetV2 with the FER dataset from Kaggle, 66% accuracy, tested on Stevens face, could not detect happy emotions **
- detecting faces & emotions on images with DeepFace ***


## Technologies
- openCV
- DeepFace
- EmoPy
- pytorch
- Tensorflow
- CNN
- MobileNetV2
- Adverserial Networks
- Facial landmarks
- Streamlit
- Microsoft Azure
- ...


## To use

To build basic CNN model trained on the FER2013 dataset use `Emo_model.py` located in the `Emo architecture CNN` folder.
```
cd EmoarchitectureCNN
python Emo_model.py
```

To build a faster & lightweight FER model for real-time detection use `detect_emotion_video.py`  This will open a webcam and give FER overlaid on the webcam stream.
```
python detect_emotion_video.py
```

To detect facial landmarks, try the jupyter notebook `EmotionRecognition.ipynb` in the `FacialLandmarks` folder.
```
cd FacialLandmarks
EmotionRecognition.ipynb
```

To generate adversarial FER images, try the jupyter notebook `Adversarial_FER.ipynb` in the `Adversarial_FER` folder


To detect emotions in realtime
```
all the file-names that start with: realtime_emotion_recognition....

(the end of the file-name describes which dataset was used to train the model)
```

To detect emotions from what is verbally said in a video
```
mp4_to_emotion.py
```

To detect emotions from photos
```
face_and_emotion_detection_images_realtime.py
```


