import cv2, time

#create object , 0 = external camera
video = cv2.VideoCapture(0)

a = 0

while True:
    a = a + 1
    #create a frame object
    check, frame = video.read()

    #print(check)
    #print(frame)

    #converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #show the frame
    cv2.imshow("capturing", gray)

    #for press any key to out (milliseconds)
    #cv2.waitKey(0)

    #for playing
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


print(a) #amount of milliseconds the streaming will take

#shutdown the camera
video.release()
cv2.destroyAllWindows