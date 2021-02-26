from picamera import PiCamera
from time import sleep
from io import BytesIO
from datetime import datetime, date, time
import numpy as np

#~ https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/
#~ https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

camera = PiCamera()
camera.rotation = -90

camera.resolution = (320, 240)
camera.framerate = 24

camera.start_preview()
camera.capture('Pictures/lastest.jpg')
camera.stop_preview()