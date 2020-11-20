## Bagaar Microexpression Detection

- Steven, Robin, Seppe, Kristof

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

Microexpressions were first discovered by researchers Haggard and Isaacs.
Dr. Paul Ekman popularized the term “microexpression” and greatly expanded the research.

## Microexpressions examples

Microexpressions you can try to detect:

Surprise ->
- The eyebrows are raised and curved.
- Skin below the brow is stretched.
- Horizontal wrinkles show across the forehead.
- Eyelids are opened, white of the eye showing above and below.
- Jaw drops open and teeth are parted but there is no tension or stretching of the mouth.

Fear ->
- Eyebrows are raised and drawn together, usually in a flat line.
- Wrinkles in the forehead are in the center between the eyebrows, not across.
- Upper eyelid is raised, but the lower lid is tense and drawn up.
- Eyes have the upper white showing, but not the lower white.
- Mouth is open and lips are slightly tensed or stretched and drawn back

Disgust ->
- Eyes are narrowed.
- Upper lip is raised.
- Upper teeth may be exposed.
- Nose is wrinkled.
- Cheeks are raised.

Anger ->
- The eyebrows are lowered and drawn together.
- Vertical lines appear between the eyebrows.
- Lower lip is tensed.
- Eyes are in hard stare or bulging.
- Lips can be pressed firmly together, with corners down, or in a square shape as if shouting.
- Nostrils may be dilated.
- The lower jaw juts out.

Happy ->
- Corners of the lips are drawn back and up.
- Mouth may or may not be parted, teeth exposed.
- A wrinkle runs from outer nose to outer lip.
- Cheeks are raised.
- Lower eyelid may show wrinkles or be tense.
- Crow’s feet near the outside of the eyes.

Sadness ->
- (It’s also one of the hardest microexpressions to correctly identify)
- (one of the longer-lasting microexpressions.)
- Inner corners of the eyebrows are drawn in and then up.
- Skin below the eyebrows is triangulated, with inner corner up.
- Corner of the lips are drawn down.
- Jaw comes up.
- Lower lip pouts out.

Contempt ->
- (it’s the only one of the 7 universal microexpressions that is asymmetrical.)
- One side of the mouth is raised.


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


