https://drive.google.com/file/d/1ee7_kKtK-Q_meOQ0jzpm1LXwk2OnPvEj/view
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

    # Save or display the residual frame
    cv2.imshow('Residual Frame', residual)
    cv2.waitKey(1)

    previous_frame = current_frame

video.release()
cv2.destroyAllWindows()
