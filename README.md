import cv2
import numpy as np

# Load the video
video = cv2.VideoCapture('video.mp4')

ret, previous_frame = video.read()

if not ret:
    print("Can't receive frame. Exiting ...")
    exit()

while True:
    ret, current_frame = video.read()

    if not ret:
        break

    # Compute residual frame
    residual = cv2.absdiff(previous_frame, current_frame)

    # Display the current frame (original video)
    cv2.imshow('Original Video', current_frame)

    # Display the residual frame
    cv2.imshow('Residual Frame', residual)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Break the loop on 'q' key press
    if key == ord("q"):
        break

    previous_frame = current_frame

video.release()
cv2.destroyAllWindows()
