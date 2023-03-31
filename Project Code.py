import cv2
import logging as log
import datetime as dt

face_casacade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_casacade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

    # loop over the detected faces
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # detects eyes of within the detected face area (roi)
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # draw a rectangle around eyes
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
