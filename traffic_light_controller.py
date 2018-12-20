import numpy as np
import cv2
import base64
import requests
import struct
import pickle
import zlib
import time
import RPi.GPIO as GPIO
import threading
from time import sleep
import sys
import socket


import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import subprocess

# display oled configuration
RST = None
disp = Adafruit_SSD1306.SSD1306_128_32(rst=RST)
disp.begin() # init display library

# init blank image
disp.clear()
disp.display()
    
oled_width = disp.width
oled_height = disp.height
image = Image.new('1',(oled_width,oled_height))

draw = ImageDraw.Draw(image)

# init display position
oled_top = -2
x = 0

# load default font
font = ImageFont.load_default()
draw.text((x,oled_top), "Hello World", font= font, fill=255)

def display_oled(nama_jalan, timee, time_default):
    # clear display before showing text
    draw.rectangle((0,0,oled_width,oled_height), outline=0,fill=0)
    
    # write txt
    draw.text((x,oled_top), "Intersection at", font= font, fill=255)
    draw.text((x,oled_top+8), nama_jalan, font=font, fill=255)
    draw.text((x,oled_top+16), "Timer Default : "+str(time_default), font=font, fill=255)
    draw.text((x,oled_top+24), "Time Remaining : "+str(timee), font=font, fill=255)
    
    #display image
    disp.image(image)
    disp.display() 

# sending socket
server_ip = '192.168.43.167'
PORT_SERVER = 8989

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.settimeout(5)

attempt1 = 0
while True:
    try:
        attempt1+=1
        client_socket.connect((server_ip,PORT_SERVER))
        connection = client_socket.makefile('wb')
    except socket.error as e:
        print("couldn't make a connection, error {}".format(e))
        
        if attempt1 == 8:
            print("too much attempts exiting")
            sys.exit(0)
        
        print("retrying connection in 2 seconds")
        sleep(2)
        continue
    else:
        break
    
def connectAttempt():
    global client_socket
    client_socket.close()
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5)
    try:
        print("retry to connect")
        client_socket.connect((server_ip,PORT_SERVER))
        print("lol")
    except socket.error as e:
        print("couldn't make a connection, error {}".format(e))
        print("continue")
        return False
    else:
        print("re-connection success")
        return True

    
# listening socket
HOST = ""
PORT = 8787

this_listen = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print("socket created")

try:
    print("binding")
    this_listen.bind((HOST,PORT))
except socket.error as e:
    print("couldn't bind socket {} ".format(PORT,e))
    print("exiting program")
    sys.exit(0)

"""
Set up servo
"""
#GPIO.setmode(GPIO.BOARD)
GPIO.setup(4, GPIO.OUT)
pwm=GPIO.PWM(4, 50)
pwm.start(0)
    
"""
INIT VARIABLE
"""
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cam = 0
timer_default = [30,30,30,30]
# timer_default = [40,40,40,40]
timer_new = 10
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


"""
PIN LAMPU
"""
M1 = 17
K1 = 27
H1 = 22

M2 = 23
K2 = 24
H2 = 25

M3 = 13
K3 = 19
H3 = 26

M4 = 16
K4 = 20
H4 = 21

GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)
GPIO.setup(M1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(K1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(H1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(M2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(K2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(H2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(M3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(K3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(H3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(M4, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(K4, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(H4, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)


    
"""
Intersection object
"""
class isc:
    M = 0
    K = 0
    H = 0
    
    def red(self):
        GPIO.output(self.M, GPIO.HIGH)
        GPIO.output(self.K, GPIO.LOW)
        GPIO.output(self.H, GPIO.LOW)
        #print("red")
    
    def yellow(self):
        GPIO.output(self.M, GPIO.LOW)
        GPIO.output(self.K, GPIO.HIGH)
        GPIO.output(self.H, GPIO.LOW)
        #print("yellow")
    
    def green(self):
        GPIO.output(self.M, GPIO.LOW)
        GPIO.output(self.K, GPIO.LOW)
        GPIO.output(self.H, GPIO.HIGH)
        #print("green")

    def __init__(self, pM, pK, pH):
        self.M = pM
        self.K = pK
        self.H = pH

isc1 = isc(M1, K1, H1)
isc1.green()
isc2 = isc(M2, K2, H2)
isc2.red()
isc3 = isc(M3, K3, H3)
isc3.red()
isc4 = isc(M4, K4, H4)
isc4.red()
next = isc(0, 0, 0)
prev = isc1


# ganti lampu ke persimpangan selanjutnya dan set timer hasil listen
# data dari server
def toggle_isc(next_isc, prev_isc, selector_timer, nama_jalan):
    next_isc.yellow()
    prev_isc.yellow()
    sleep(3)
    #you can set new timer here gan!, setelah lu dapet nilai dari
    #server, you bandingin sama timer sekarang yang di array
    next_isc.green()
    prev_isc.red()
    if timer_new > 0:
        timer_use = timer_new
        print("new {} use {}".format(timer_new, timer_use))
    else:
        timer_use = timer_default[selector_timer]
        
    while timer_use > 0:
        sleep(1)
        print(timer_use)
        display_oled(nama_jalan, timer_use, timer_default[selector_timer])
        timer_use -= 1
    

# function capture camera
def Capture():
    if cam == 1:
        cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        r, frame = cap1.read()
    elif cam == 2:
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        r, frame = cap2.read()
    else:
        print("Camera can't work")
    
    if r:
        nama_file = "cam_"+str(cam)
        print(nama_file)
        r, buffer = cv2.imencode(".jpg",frame, encode_param)
        print(buffer)
            
        data = pickle.dumps(buffer,0)
        size = len(data)
        
        print("{}: {}".format(cam, size))
        try:
            client_socket.sendall(struct.pack(">L", size) + data)
        except socket.error as e:
            print("capture error, {}".format(e))
            if connectAttempt():
                client_socket.sendall(struct.pack(">L", size) + data)
                return
            else:
                return
            
           
def listen():
    global timer_new
    
    try:
        this_listen.settimeout(15)
        this_listen.listen(10)
        
        print("listening")
    
        conn,addr = this_listen.accept()
    
        print("accepting")    
        data = conn.recv(1024)
    except socket.error as e:
        print("connection timeout {}".format(e))
        print("set time to default")
        timer_new = 0
        return
    else:
        print("the data is {}".format(data))
        # timer_new = int.from_bytes(data,byteorder = 'little')
        timer_new = int(struct.unpack("B", data)[0])
        print("timer is {}".format(timer_new))
        # this_listen.close()

# set servo angle
def SetAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(4, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(4, False)
    pwm.ChangeDutyCycle(0)


#ignition engine, calibration
SetAngle(90)
#cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap1.read()
#cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap2.read()
sleep(2)
print("engine on")
print("----START----")

while True:
    print('intersection 1 green')
    cam=1
    Capture()
    #di function ini dapet selector intersection mana yang nyala, dan
    #timer ke berapa yang harusnyala  (diliat dari array)
    listen()
    sleep(1)
    toggle_isc(isc2, isc1, 0, "KFC 1")
    print('intersection 1 red')
    print('move')
    SetAngle(20)
    
    print('intersection 2 green')
    cam=1
    Capture()
    listen()
    sleep(1)
    toggle_isc(isc3, isc2, 1, "KFC 2")
    print('intersection 2 red')
    print('move')
    SetAngle(90)
    
    print('intersection 3 green')
    cam=2
    Capture()
    listen()
    sleep(1)
    toggle_isc(isc4, isc3, 2, "KFC 3")
    print('intersection 3 red')
    print('move')
    SetAngle(20)
    
    print('intersection 4 green')
    cam=2
    Capture()
    listen()
    sleep(1)
    toggle_isc(isc1, isc4, 3, "KFC 4")
    print('intersection 4 red')
    print('move')
    SetAngle(90)
    
    