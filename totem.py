import RPi.GPIO as GPIO
from picamera import PiCamera
import picamera.array
from time import sleep
from io import BytesIO
from datetime import datetime, date#, time
import time, uptime, argparse

URL="https://myconnecapp.fr/public/totem.php"
SITE="home"

#~ https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/
#~ https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
#~ from imutils.perspective import four_point_transform
#~ from imutils import contours
#~ import imutils
import cv2

#~ my_stream = BytesIO()

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="get image", action="store_true")
parser.add_argument("-m", help="play music", action="store_true")
args = parser.parse_args()

V = 40 # 120
avgM = 0.75 # 0.5 si 'Lo', 1.0 si autre !
rotation = 0
# *** CROP ***
rV = (230,340) #(160,470) 0-480
rH = (290,400) #(200,500) 0-720

if args.m:
    print("Play...")
    for i in range(0,10):
            pygame.mixer.init()
            pygame.mixer.music.load("Music/num-%d.mp3" % i)
            pygame.mixer.music.set_volume(2.0)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                pass
            print("Done !")
    exit()

import numpy as np, io

if args.i:
    with picamera.PiCamera() as camera:
        camera.rotation = rotation
        #camera.exposure_mode = 'off'
        #camera.awb_mode = 'off'
        #camera.awb_gains = 0.9 # Red avg ~ 71, 1.0 avg 40
        camera.start_preview()
        time.sleep(30)
        with picamera.array.PiRGBArray(camera) as output:
            camera.capture(output, 'rgb')
            img = output.array
            print(img.shape) # (480, 720, 3)
            (R, G, B) = np.average(img, axis=(0,1))
            print(R, G, B)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("Pictures/capture.jpg", img)
            img = img[rV[0]:rV[1],rH[0]:rH[1],:]
            cv2.imwrite("Pictures/capture-crop.jpg", img)
            if (G>5):
                img = img[:,:,1]
                print(img.shape)
                cv2.imwrite("Pictures/capture-color.jpg", img)
        print("Capture in Pictures/capture.jpg")
        exit()

import ssocr, pygame, requests
        

f = open('capteur.log', 'a')

# LEDs red/green
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT) # LED rouge
GPIO.setup(3, GPIO.OUT) # LED verte

t0=time.time()
t=0
with picamera.PiCamera() as camera:
    #~ camera.shutter_speed = 50 # camera.exposure_speed
    print("exposure_speed", camera.exposure_speed)
    camera.rotation = rotation
    camera.exposure_mode = 'off'
    camera.awb_mode = 'off'
    camera.awb_gains = 1.0 # Red avg ~ 71, 1.0 avg 40
    with picamera.array.PiRGBArray(camera) as output:
        #~ camera.resolution = (320, 240)
        while True:
        #~ for i in range(100):
                t0 = t
                t = time.time()
                camera.capture(output, 'rgb')
                img0 = output.array
                #~ print(img.shape) # (480, 720, 3)
                
                #~ img = img0[150:400,180:479,:] # *** CROP ***
                img = img0[rV[0]:rV[1],rH[0]:rH[1],:] # *** CROP ***
                
                GPIO.output(2, GPIO.LOW) # Eteind la rouge
                GPIO.output(3, GPIO.LOW) # Eteind la verte
                (R, G, B) = np.average(img, axis=(0,1)) # (np.average(img[:,:,0]), np.average(img[:,:,1]), np.average(img[:,:,2]))
                avg = 0.0
                sum=0.0
                msg=''
                
                #~ if G>V: img = img[:,:,1]
                #~ if R>V: img = img[:,:,0]
                
                if G>V or R>V:
                        #OCR
                        #~ print('OCR')
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        #~ print("RGB", R, G, B)
                        #~ cv2.imwrite("image-raw.jpg", cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))
                        #~ cv2.imwrite("image-1.jpg", img)
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        #~ cv2.imwrite("image-gray.jpg", gray_img)
                        blurred, gray_img = ssocr.load_image(gray_img=gray_img)
                        outp = blurred
                        #~ cv2.imwrite("image-blurred.jpg", blurred)
                        dst = ssocr.preprocess(blurred, ssocr.THRESHOLD)
                        digits_positions = ssocr.find_digits_positions(dst)
                        #~ print("digits_positions: ", digits_positions)
                        digits = ssocr.recognize_digits_line_method(digits_positions, outp, dst)                
                        msg = digits
                        temp = 0.0
                        try:
                                temp=int(digits)/10.0
                        except:
                                temp=0.0 #continue
                        
                        if G>V:
                                GPIO.output(3, GPIO.HIGH) # Alume la verte
                                #~ msg='GREEN'
                                cv2.imwrite("Pictures/green.jpg", img)
                                #~ requests.get(URL, {"site": SITE, "status": "Green", "temp": temp, "raw": digits}) 
                        if R>V and msg!="Lo" and msg!="*" and msg!="" and msg!="L":
                                GPIO.output(2, GPIO.HIGH) # Alume la rouge
                                cv2.imwrite("Pictures/red.jpg", img)
                                #~ requests.get(URL, {"site": SITE, "status": "Red", "temp": temp, "raw": digits}) 
                                #~ img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                #~ msg = 'HOT !'
                                
                                #~ cv2.imwrite("Pictures/red.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                                '''
                                res = np.bitwise_xor(img2, Lo);
                                avg = np.average(res)
                                if avg > avgM: 
                                        GPIO.output(2, GPIO.HIGH) # Alume la rouge
                                        msg = 'HOT !'
                                        cv2.imwrite("Pictures/red.jpg", img)
                                #~ print("                         avg", avg)
                                if False:
                                        cv2.imwrite("Pictures/red.jpg", img)
                                        cv2.imwrite("Pictures/redBW.jpg", img2)
                                        res = np.bitwise_xor(img2, Lo);
                                        cv2.imwrite("Pictures/redXOR.jpg", res)
                                        avg = np.average(res)
                                        sum = mp.sum(res)
                                        #~ print("saved, avg=", avg)
                                #~ sleep(4)
                                '''
                txt = '%.2f R%5.1f G%5.1f B%5.1f - %s' % (uptime.uptime(), R, G, B, msg)
                print(txt)
                f.write(txt+"\n")
                output.truncate(0)
                

GPIO.output(2, GPIO.LOW) # Eteind la rouge
GPIO.output(3, GPIO.LOW) # Eteind la verte
exit()


now = datetime.now().strftime("%Y-%m-%d %X")
print(now)

camera = PiCamera()
camera.rotation = -90
#~ camera.resolution = (128, 128)
#camera.annotate_text = "Hello world!"
#camera.annotate_text_size = 50
#camera.image_effect = 'cartoon'

#~ camera.start_recording('rpi%s.h264' % now)
#~ camera.wait_recording(60)
#~ camera.stop_recording()
#~ exit()
camera.resolution = (320, 240)
camera.framerate = 24

sleep(2)
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
print(g)
camera.awb_mode = 'off'
camera.awb_gains = 1

stream = io.BytesIO()
stream = picamera.array.PiRGBArray(camera)


for i in range(50):
        print("loop %d"% i)
        stream = camera.capture(stream, format='bgr')
        #~ image = stream.array
        #~ camera.capture(stream, format='jpeg')
        #~ camera.capture('Pictures/rpi-%2.2d.jpg' % (i))


exit()

camera.start_preview()
for i in range(50):
        #~ print("i=", i)
        #~ camera.capture('Pictures/rpi %s-%2.2d.jpg' % (now, i))
        #~ output = np.empty((240, 320, 3), dtype=np.uint8)
        #~ output = np.empty((1920, 1080, 3), dtype=np.uint8)
        output = np.empty((320 * 240 * 3,), dtype=np.uint8)
        camera.resolution = (320, 240)
        camera.capture('Pictures/rpi-%2.2d.jpg' % (i))
        camera.framerate = 24
        camera.capture(output, 'rgb')
        output = output.reshape((320, 240, 3))
        #~ print "avg: %f" % (np.average(output))
        (R, G, B) = (np.average(output[:,:,0]), np.average(output[:,:,1]), np.average(output[:,:,2]))
        x = ""
        if R>V: x = "RED"
        if G>V: x = "GREEN"
        print("%-2d %5s: %5.1f %5.1f %5.1f " % (i, x, R, G, B))
        #~ print output[:,:,1]
        #~ sleep(0.5)

#~ camera.capture('Pictures/foo.jpg')
#~ sleep(20)
camera.stop_preview()
exit()

while (True):
    camera.start_preview(alpha=250)
    sleep(20)
    camera.stop_preview()
