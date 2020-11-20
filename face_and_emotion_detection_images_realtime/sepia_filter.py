#this code will create a sepia filter so you are harder to recognise for models to analyse you
# (for protectoin against privacy)

import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import time

vs = VideoStream(0).start()
time.sleep(2.0)

# Create sepia filter
kernel = np.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]])

# loop over the frames from the video stream
while True:

    # grab the frame from the threaded video stream and resize it
    frame = vs.read()

    if frame is not None:
        frame = imutils.resize(frame, 400)

    # Apply sepia filter
    output = cv2.filter2D(frame, -1, kernel)
    cv2.imshow("Facial Expression", output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()