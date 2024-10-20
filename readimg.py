import time 



import os
import io
import cv2
from threading import Thread
from PIL import Image
import pyttsx3
engine = pyttsx3.init()
import easyocr

reader=easyocr.Reader(['en'])
# thread function to convert string to audio

# function to convert image to text
def img_txt(image):
    result = reader.readtext(r'C:/Users/bvdis/blindassist/blindassist/captured_image.jpg', detail=0)
    resultstr = "".join(result)
    print('audio begin')

    engine.say(resultstr)
    engine.runAndWait()
    print(resultstr)


# Open the camera
camera = cv2.VideoCapture(0)

# Allow camera to warm up
time.sleep(2)

# Capture the image
ret, frame = camera.read()

# Release the camera
camera.release()

# Save the captured image to a file
cv2.imwrite("captured_image.jpg", frame)

# Read the captured image using cv2.imread
captured_image = cv2.imread("captured_image.jpg")


# Convert the resized image to text
img_txt(captured_image)