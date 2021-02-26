from picamera import PiCamera
from time import sleep
from io import BytesIO
from datetime import datetime, date, time
import numpy as np
import os
#~ from picamraw import PiRawBayer, PiCameraVersion
from imutils.perspective import four_point_transform
import numpy as np
import imutils
import cv2

V = 120
path = 'Pictures/'
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

(xmin, xmax, ymin, ymax) = (313, 436,  157, 256)
xdelta = xmax-xmin
ydelta = ymax-ymin
x3 = int(xdelta/3) # 3 digits

ref = {}
for n in list('0123456789lo'):
	#~ print(n)
	img = cv2.imread("num-%s.jpg"%n)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ref[n] = img


no=0
def chkFile(fn):
	img = cv2.imread(path+fn)
	color = 'BLACK'
	global no
	no+=1
	(B, G, R) = (np.average(img[:,:,0]), np.average(img[:,:,1]), np.average(img[:,:,2]))
	if R>V: color = "RED"
	if G>V: color = "GREEN"
	print("%2d %-40s %5s %s %5.1f %5.1f %5.1f " % (no, fn, color, img.shape, R, G, B))
	
	image = imutils.resize(img, height=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	'''
	#~ cv2.imwrite("Pictures/%d-gray.jpg"%no, gray)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	#~ cv2.imwrite("Pictures/%d-blur.jpg"%no, blurred)
	edged = cv2.Canny(blurred, 50, 200, 255)
	#~ cv2.imwrite("Pictures/%d-edged.jpg"%no, edged)

	# find contours in the edge map, then sort them by their
	# size in descending order
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	displayCnt = None
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if the contour has four vertices, then we have found
		# the thermostat display
		if len(approx) == 4:
			displayCnt = approx
			break
	# extract the thermostat display, apply a perspective transform
	# to it
	warped = four_point_transform(gray, displayCnt.reshape(4, 2))
	output = four_point_transform(image, displayCnt.reshape(4, 2))
	'''
	#~ pts = np.array([(313,254), (319,157), (436,157), (429,256)])
	pts = np.array([(xmin,ymax), (xmin,ymin), (xmax,ymin), (xmax,ymax)])
	output = four_point_transform(gray, pts)
	#~ cv2.imwrite("Pictures/%d-image.jpg"%no, image)
	cv2.imwrite("Pictures/%d-output.jpg"%no, output)
	
	# threshold the warped image, then apply a series of morphological
	# operations to cleanup the thresholded image
	v,thresh = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	cv2.imwrite("Pictures/%d-thresh.jpg"%no, thresh)
	print("thres=", v, thresh.shape)
	'''
	v,thresh = cv2.threshold(output, 210, 255, cv2.THRESH_BINARY_INV )
	cv2.imwrite("Pictures/%d-thresh1.jpg"%no, thresh)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	cv2.imwrite("Pictures/%d-thresh2.jpg"%no, thresh)
	th3 = cv2.adaptiveThreshold(output, 160, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
	cv2.imwrite("Pictures/%d-thresh2.jpg"%no, thresh)
	th3 = cv2.adaptiveThreshold(output, 160, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
	cv2.imwrite("Pictures/%d-thresh3.jpg"%no, thresh)
	'''
	for i in range(3):
		#~ print(i)
		digit = thresh[:, i*x3:x3+i*x3]
		cv2.imwrite("Pictures/%d-digit%s.jpg"%(no,i), digit)
		for j in ref.keys():
			res = np.bitwise_xor(digit, ref[j]);
			avg = np.average(res)
			if avg<50: 
				print("digit %d / %s: avg %f"%(i, j, avg))
				cv2.imwrite("Pictures/%d-digit%s-%s.jpg"%(no,i,j), res)
	#~ cv2.imwrite("Pictures/%d-digit1.jpg"%no, thresh[:, :x3])
	#~ cv2.imwrite("Pictures/%d-digit2.jpg"%no, thresh[:, x3:x3*2])
	#~ cv2.imwrite("Pictures/%d-digit3.jpg"%no, thresh[:, x3*2:])
	


for fn in os.listdir(path+"green"): chkFile("green/"+fn)
for fn in os.listdir(path+"red"): chkFile("red/"+fn)
for fn in os.listdir(path+"black"): chkFile("black/"+fn)
