#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import json
import pickle
import time
import cv2
import random
import math
import socket
import sys
import argparse
import sys

from datetime import datetime

import tensorflow as tf
from tensorflow.python.client import device_lib

import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  matplotlib import pyplot

import keras
import keras.backend as K
from keras.datasets import imdb
from keras.models import Sequential,Model,model_from_json
from keras.layers import Input,Dense,Lambda,Reshape,RepeatVector,Dot,merge,Concatenate,Add,Dropout,Conv2D,MaxPooling2D,Flatten,Conv1D,Softmax
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D,MaxPooling2D,Flatten,Conv1D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.preprocessing import image as image_p
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.activations import softmax,tanh

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import WeightReader, decode_netout, draw_boxes

from PIL import Image

device_lib.list_local_devices()

def fitEllipse2(xy):
    x = np.asarray(xy[:,0])
    y = np.asarray(xy[:,1])
    D1 =  np.hstack((x*x, x*y, y*y))
    D2 =  np.hstack((x, y, np.ones_like(x)))
    S1 = np.dot(D1.T,D1)
    S2 = np.dot(D1.T,D2)
    S3 = np.dot(D2.T,D2)
    iS3 = np.linalg.inv(S3)
    T = np.dot(-iS3, S2.T)
    M = S1 + np.dot(S2, T)
    C = np.zeros([3,3])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(C), M))

    for i in range(3):
        cond = 4*V[0,i]*V[2,i] - V[1,i]*V[1,i]
        if cond > 0:
            n = i
            break

    a = V[:,n]
    a = np.append(a, np.dot(T,a))
    return a

def ellipseToCircle(a, f, r):
    f = camera_matrix[0,0]
    A = np.array([[a[0], a[1]/2, -a[3]/(2*f)], [a[1]/2, a[2], -a[4]/(2*f)], [-a[3]/(2*f), -a[4]/(2*f), a[5]/(f*f)]])
    E, V =  np.linalg.eig(A)
    V = np.asmatrix(V)
    la = E[0]
    lb = E[1]
    lc = E[2]

    if np.sign(la) == np.sign(lb):
        l3 = lc
        three = 2;
        if abs(la) > abs(lb):
            l1 = la
            one = 0
            l2 = lb
            two = 1
        else:
            l1 = lb
            one = 1
            l2 = la
            two = 0         
    
    elif np.sign(la) == np.sign(lc):
        l3 = lb
        three = 1
        if abs(la) > abs(lc):
            l1 = la
            one = 0
            l2 = lc
            two = 2
        else:
            l1 = lc
            one = 2
            l2 = la
            two = 0           

    elif np.sign(lb) == np.sign(lc):
        l3 = la
        three = 0
        if abs(lb) > abs(lc):
            l1 = lb
            one = 1
            l2 = lc
            two = 2
        else:
            l1 = lc
            one = 2
            l2 = lb
            two = 1           

    V = np.matrix([[V[0,one], V[0,two], V[0,three]], [V[1,one], V[1,two], V[1,three]], [V[2,one], V[2,two], V[2,three]]])

    Cc = []
    Nc = []
    for s1 in range(-1, 2,2):
        for s2 in range(-1, 2,2):
            for s3 in range(-1, 2,2):
                z0 = (s3*l2*r/math.sqrt(-l1*l3))
                Ci = np.array([[s2*l3/l2*math.sqrt((l1-l2)/(l1-l3))], [0], [-s1*l1/l2*math.sqrt((l2-l3)/(l1-l3))]])
                Ci = z0*np.dot(V,Ci)
                Ni = np.array([[s2*math.sqrt((l1-l2)/(l1-l3))], [0], [-s1*math.sqrt((l2-l3)/(l1-l3))]])
                Ni = np.dot(V,Ni) 
                if Ci[2] < 0  and Ni[2] > 0:
                    Cc.append(Ci)
                    Nc.append(Ni)
    return Cc, Nc

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.degrees(np.array([x, y, z]))


def process_image_keypoints(img,bbox_coords):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    bbox_coordinates = []
    for a,b in bbox_coords:
        a = float(a)*ratio
        b = float(b)*ratio
        bbox_coordinates.append([a+left,b+top])
    return new_im,bbox_coordinates


# In[6]:


def process_image_keypoints_nobox(img):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im,[left,top,ratio]

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
            
class Message:
    def __init__(self):    
        self.message_type = "001"
        self.date = ""
        self.time = ""
        self.target_range = -10
        self.target_range_valid = 0
        self.target_heading = 0
        self.target_heading_valid = 0
        self.target_elevation = 0
        self.target_elevation_valid = 0
        self.target_quality = 1
        self.number_of_lights = 0
        self.number_of_col_lights = 0
        self.lighting_case = 0
        self.detection_mode = 1
        self.camera_status = 0
        self.checksum = ""
    def set_target_detected(self,target_quality):
        self.target_quality = target_quality    
    def set_number_of_lights(self,num_lights, num_col_lights):
        self.number_of_lights = num_lights
        self.number_of_col_lights = num_col_lights 
    def set_camera_status(self,success):
    	if success==True:
    		self.camera_status = 1
    	else:
    	 	self.camera_status = -1        
    def convert_to_string(self):
    	return ",".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,"{0:.1f}".format(self.target_range),str(self.target_range_valid),"{0:.1f}".format(self.target_heading),str(self.target_heading_valid),"{0:.1f}".format(self.target_elevation),str(self.target_elevation_valid),str(self.target_quality),str(self.number_of_lights),str(self.number_of_col_lights),str(self.lighting_case),str(self.detection_mode),str(self.camera_status),self.checksum])	
    def set_date_time(self):
    	self.date = datetime.now().strftime('%Y%m%d')
    	self.time = datetime.now().strftime('%H:%M:%S.%f')[:-4]
    def set_lighting_case(self, lighting_case):
        self.lighting_case = lighting_case
    def fill_target_heading_elevation(self,distance,heading,elevation):
    	self.target_range = distance
    	self.target_heading =  heading
    	self.target_elevation =  elevation
    def fill_target_valid(self,distance,heading,elevation):
    	self.target_range_valid = distance
    	self.target_heading_valid =  heading
    	self.target_elevation_valid =  elevation
    def set_checksum(self):
    	chk_sum_string = "".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,"{0:.1f}".format(self.target_range),str(self.target_range_valid),"{0:.1f}".format(self.target_heading),str(self.target_heading_valid),"{0:.1f}".format(self.target_elevation),str(self.target_elevation_valid),str(self.target_quality),str(self.number_of_lights),str(self.number_of_col_lights),str(self.lighting_case),str(self.detection_mode),str(self.camera_status),self.checksum])	
    	calc_cksum = 0
    	for s in chk_sum_string:
    	    	calc_cksum ^= ord(s)
    	self.checksum = str(hex(calc_cksum)).lstrip("0").lstrip("x")

class Message2:
    def __init__(self):    
        self.message_type = "002"
        self.date = ""
        self.time = ""
        self.number_of_lights = 0
        self.number_of_col_lights = 0
        self.light1_x = 0
        self.light1_y = 0
        self.light2_x = 0
        self.light2_y = 0
        self.light3_x = 0
        self.light3_y = 0
        self.light4_x = 0
        self.light4_y = 0
        self.light5_x = 0
        self.light5_y = 0
        self.light6_x = 0
        self.light6_y = 0
        self.light7_x = 0
        self.light7_y = 0
        self.light8_x = 0
        self.light8_y = 0
        self.light9_x = 0
        self.light9_y = 0
        self.light1_color_x = 0
        self.light1_color_y = 0
        self.light2_color_x = 0
        self.light2_color_y = 0
        self.camera_status = 0
        self.checksum = ""
    def set_number_of_lights(self,num_lights, num_col_lights):
        self.number_of_lights = num_lights
        self.number_of_col_lights = num_col_lights                 
    def set_camera_status(self,success):
    	if success==True:
    		self.camera_status = 1
    	else:
    	 	self.camera_status = -1
    def fill_light_positions(self,image_points):
    	self.number_lights = len(image_points)
    	if len(image_points)>=1:
    		self.light1_x = image_points[0][0][0]
    		self.light1_y = image_points[0][1][0]
    	if len(image_points)>=2:
    		self.light2_x = image_points[1][0][0]
    		self.light2_y = image_points[1][1][0]
    	if len(image_points)>=3:
    		self.light3_x = image_points[2][0][0]
    		self.light3_y = image_points[2][1][0]
    	if len(image_points)>=4:
    		self.light4_x = image_points[3][0][0]
    		self.light4_y = image_points[3][1][0]
    	if len(image_points)>=5:
    		self.light5_x = image_points[4][0][0]
    		self.light5_y = image_points[4][1][0]
    	if len(image_points)>=6:
    		self.light6_x = image_points[5][0][0]
    		self.light6_y = image_points[5][1][0]
    	if len(image_points)>=7:
    		self.light7_x = image_points[6][0][0]
    		self.light7_y = image_points[6][1][0]
    	if len(image_points)>=8:
    		self.light8_x = image_points[7][0][0]
    		self.light8_y = image_points[7][1][0]
    	if len(image_points)>=9:
    		self.light9_x = image_points[8][0][0]
    		self.light9_y = image_points[8][1][0] 
    def fill_color_light_position(self,image_points):
        self.number_lights = len(image_points)
        if len(image_points)>=1:
                self.light1_color_x = image_points[0][0][0]
                self.light1_color_y = image_points[0][1][0]
        if len(image_points)>=2:
                self.light2_color_x = image_points[1][0][0]
                self.light2_color_y = image_points[1][1][0]
    def convert_to_string(self):
        print(self.message_type)
        return ",".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,str(self.number_of_lights),str(self.number_of_col_lights),"{0:.1f}".format(self.light1_x),"{0:.1f}".format(self.light1_y),"{0:.1f}".format(self.light2_x),"{0:.1f}".format(self.light2_y),"{0:.1f}".format(self.light3_x),"{0:.1f}".format(self.light3_y),"{0:.1f}".format(self.light4_x),"{0:.1f}".format(self.light4_y),"{0:.1f}".format(self.light5_x),"{0:.1f}".format(self.light5_y),"{0:.1f}".format(self.light6_x),"{0:.1f}".format(self.light6_y),"{0:.1f}".format(self.light7_x),"{0:.1f}".format(self.light7_y),"{0:.1f}".format(self.light8_x),"{0:.1f}".format(self.light8_y),"{0:.1f}".format(self.light9_x),"{0:.1f}".format(self.light9_y),"{0:.1f}".format(self.light1_color_x),"{0:.1f}".format(self.light1_color_y),"{0:.1f}".format(self.light2_color_x),"{0:.1f}".format(self.light2_color_y),str(self.camera_status),str(self.checksum)])
    def set_date_time(self):
    	self.date = datetime.now().strftime('%Y%m%d')
    	self.time = datetime.now().strftime('%H:%M:%S.%f')[:-4]	        
    def set_checksum(self):
    	chk_sum_string = "".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,str(self.number_of_lights),str(self.number_of_col_lights),"{0:.1f}".format(self.light1_x),"{0:.1f}".format(self.light1_y),"{0:.1f}".format(self.light2_x),"{0:.1f}".format(self.light2_y),"{0:.1f}".format(self.light3_x),"{0:.1f}".format(self.light3_y),"{0:.1f}".format(self.light4_x),"{0:.1f}".format(self.light4_y),"{0:.1f}".format(self.light5_x),"{0:.1f}".format(self.light5_y),"{0:.1f}".format(self.light6_x),"{0:.1f}".format(self.light6_y),"{0:.1f}".format(self.light7_x),"{0:.1f}".format(self.light7_y),"{0:.1f}".format(self.light8_x),"{0:.1f}".format(self.light8_y),"{0:.1f}".format(self.light9_x),"{0:.1f}".format(self.light9_y),"{0:.1f}".format(self.light1_color_x),"{0:.1f}".format(self.light1_color_y),"{0:.1f}".format(self.light2_color_x),"{0:.1f}".format(self.light2_color_y),str(self.camera_status),str(self.checksum)])
    	calc_cksum = 0
    	for s in chk_sum_string:
    	    	calc_cksum ^= ord(s)
    	self.checksum = str(hex(calc_cksum)).lstrip("0").lstrip("x")    


print("----------------------------------------------------------------------------------------------------")


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weight", type=str, required=True,help="weights")
ap.add_argument("-cfg", "--config", type=str, required=True,help="configuration")
ap.add_argument("-s", "--size", type=str, required=True,help="size of light ring in m")
ap.add_argument("-b", "--blue", type=str, required=True,help="blue light separation in m")
ap.add_argument("-src", "--source", type=str, required=False,choices=['video','stream'],help="video or stream")
ap.add_argument("-l", "--lights", type=str, required=False,choices=['True','False'],help="show light coordinates or not")
ap.add_argument("-save_train", "--save_train", type=str, required=False,choices=['True','False'],help="save data to folder for training")
ap.add_argument("-save_video", "--save_video", type=str, required=False,choices=['True','False'],help="save data to folder for creating video")
ap.add_argument("-skip", "--skip", type=str, required=False,help="process every nth frame")
ap.add_argument("-video", "--video", type=str, required=False,help="video for processing")
ap.add_argument("-slink", "--stream_link", type=str, required=False,help="stream link")
ap.add_argument("-m", "--medium", type=str, required=False,choices=['air','water'],help="choose the camera medium")
ap.add_argument("-p1", "--port1", type=str, required=False,help="port 1")
ap.add_argument("-p2", "--port2", type=str, required=False,help="port 2")

args = vars(ap.parse_args())
weights_path = args["weight"]
config_path = args["config"]
hoop_size = args["size"]
col_dist = args["blue"]
source =  "stream" if args["source"]==None else args["source"]
show_lights = True if args["lights"]=='True' else False
multiplier = 5 if args["skip"]==None else int(args["skip"])
save_data_for_training = True if args["save_train"]=='True' else False
save_data_for_video = True if args["save_video"]=='True' else False     
video_link =  "stream2.mp4" if args["video"]==None else args["video"]
default_stream_link = 'rtsp://admin:admin@192.168.214.40/h264.sdp?res=half&x0=0&y0=0&x1=1920&y1=1080&qp=16&doublescan=0&ssn=41645' 
stream_link = default_stream_link if args["stream_link"]==None else args["stream_link"]
medium =  "air" if args["medium"]==None else args["medium"]
port1 =  "51110" if args["port1"]==None else args["port1"]
port2 =  "51120" if args["port2"]==None else args["port2"]

print("Weights path is ",weights_path)
print("Config path is ",config_path)
print("Light ring diameter is ",hoop_size)
print("Blue light distance is ",col_dist)
print("Source is",source)
print("show lights is",show_lights)
print("Processing every nth frame",multiplier)
print(args)


server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.settimeout(0.2)


LABELS = ['station']
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
#ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
#ANCHORS          = [0.1,0.1,0.7,0.7,2,2,5,5,9,9]
ANCHORS          = [1.05, 1.68, 1.35, 20.6, 1.50, 2.40, 1.85, 2.82] #anchors determined by k.means clustering
TRUE_BOX_BUFFER  = 10


if source=="stream":
        vidcap = cv2.VideoCapture(stream_link)
else:
        vidcap = cv2.VideoCapture(video_link)


radius = float(hoop_size)/2


with open(config_path,'rb') as f:
    cfg = pickle.load(f)
    model = model_from_json(cfg)
    

with open(weights_path,'rb') as f:
    weights = pickle.load(f)
    model.set_weights(weights)

seconds = 0.5
#fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second



success,image = vidcap.read()
#image_clean = cv2.imread('img.jpg')
#success = True
path = "stream2_images"
print(path)
try:
    os.mkdir(path);
except:
    print("exits")

img_count=0
detect_count=0
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

while success:
    #image = image_clean
    #start = time.time()
    m1 = Message()
    m2 = Message2()

    try:
        frameId = int(round(vidcap.get(1)))
        #if not frameId % multiplier==0:
        #        raise ValueError('Skipping frame id',str(frameId))
        img_count += 1

        stamp = int(time.time())        
        if save_data_for_training==True:
            cv2.imwrite("stream_training/"+ str(stamp)+".jpg",image)
                
        input_image = cv2.resize(image, (224, 224))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)

        netout = model.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0],obj_threshold=0.6,nms_threshold=0.05,anchors=ANCHORS,nb_class=CLASS)            
        
        max_score = -1
        saved_box = None

        for bbox in boxes:
                if bbox.get_score()>max_score:
                        saved_box = bbox
        image_h, image_w, _ = image.shape  
 
        if saved_box == None:
                m1.set_target_detected(0)
                img = image
                logger.info('Target not detected')
                print('No Target')        
        else:
                xmin = int(saved_box.xmin*image_w)
                ymin = int(saved_box.ymin*image_h)
                xmax = int(saved_box.xmax*image_w)
                ymax = int(saved_box.ymax*image_h)            
                
                pt1 = (int(xmin),int(ymin))
                pt2 = (int(xmax),int(ymax))
                
                img = image[int(ymin):int(ymax),int(xmin):int(xmax),:]
                for box in boxes:
                    print('-------------------')
                    print(box.get_score())
                    print(box.xmin, box.ymin)
                    print(box.xmax, box.ymax)                    
                image = draw_boxes(image, boxes, LABELS)
                detect_count += 1

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        gray_image = cv2.blur(gray_image, (10,10)) 
        max_int = np.max(gray_image)
        
        """
        for i in range (10, 110, 50):
 
                ret, th1 = cv2.threshold(gray_image,(max_int - i), 255, cv2.THRESH_BINARY)                                        #80
                ret,contours,hierachy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 4:
                        break
                        
        color_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        
        low_blue = np.array([80, 0, 100])
        high_blue = np.array([94, 255, 255])
        blue_mask = cv2.inRange(color_image, low_blue, high_blue)
        blur = cv2.blur(blue_mask, (10,10))
        max_int = np.max(blur)
        ret, th1 = cv2.threshold(blur,(max_int - 100), 255, cv2.THRESH_BINARY) 
        
        ret,contours_col,hierachy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        dist_points = np.empty((0 ,2), dtype = np.float64)       
        col_points = np.empty((0 ,2), dtype = np.float64)       
        contour_count = 0
        col_contour_count = 0
        exception_count = 0             
        for c in contours:
            try:
                col = 0
                M = cv2.moments(c)

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                for cc in contours_col:
                    try:
                        Mc = cv2.moments(cc)

                        ccX = int(Mc["m10"] / Mc["m00"])
                        ccY = int(Mc["m01"] / Mc["m00"])
                        
                        if abs(cX-ccX) < 7 and abs(cY-ccY) < 7:
                            if saved_box!=None:
                                col_points = np.append( col_points, np.array([[cX+int(xmin),cY+int(ymin)]]), axis = 0)
                            else:
                                col_points = np.append( col_points, np.array([[cX,cY]]), axis = 0)
                            col = 1
                            col_contour_count = col_contour_count + 1
                            break
                    except:
                        continue

                if col==0:
                    if saved_box!=None:
                        dist_points = np.append( dist_points, np.array([[cX+int(xmin),cY+int(ymin)]]), axis = 0)
                    else:
                        dist_points = np.append( dist_points, np.array([[cX,cY]]), axis = 0)
                    contour_count = contour_count + 1
            except:
                exception_count = exception_count + 1  
                
        # Camera internals
        focal_length = 768.31
        center = (479.72, 104.56)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double")
        distortion_coeffs = np.array([[-0.3023023,0.14315257,-0.00201115,-0.00041268,-0.04129913]])
                                        
        lighting_case = 0
                
        if contour_count > 4:                                                # hoop fit
            logger.info("Hoop Fit")
            print("Hoop Fit Detected")
            
            dist_points_r = dist_points.reshape(contour_count, 1, 2)
            undist_points = cv2.undistortPoints(dist_points_r, camera_matrix, distortion_coeffs)
            xy = np.asmatrix(undist_points*focal_length)    

            a = fitEllipse2(xy) 
            Cc, Nc = ellipseToCircle(a, focal_length, radius)
            translation_vector = [[float(-(Cc[0][0]+Cc[1][0])/2)], [float(-(Cc[0][1]+Cc[1][1])/2)], [float(-(Cc[0][2]+Cc[1][2])/2)]]
            
            light_coord = np.array([undist_points*float(translation_vector[2][0])])
            m2.fill_light_positions(light_coord)            
            
            lighting_case = 3
            color = (0,255,0)

            if col_contour_count == 2: 
                print("Blue Lights")
                col_points_r = col_points.reshape(col_contour_count, 1, 2)
                undist_col_points = cv2.undistortPoints(col_points_r, camera_matrix, distortion_coeffs)
               
                col_dist = np.linalg.norm(undist_col_points)*float(translation_vector[2][0])/focal_length
                color_coord = np.array([undist_col_points*float(translation_vector[2][0])])
                m2.fill_color_light_position(color_coord)
            
        elif col_contour_count == 2:
            logger.info("Color Light Fit")
        
            col_points_r = col_points.reshape(col_contour_count, 1, 2)
            undist_col_points = cv2.undistortPoints(col_points_r, camera_matrix, distortion_coeffs)
            xy = np.asmatrix(undist_col_points*focal_length)  
                        
            z = focal_length*col_dist/np.linalg.norm(xy)
            x = (xy[0,0] + xy[0,1])/2*col_dist/np.linalg.norm(xy)
            y = (xy[1,0] + xy[1,1])/2*col_dist/np.linalg.norm(xy)
            
            translation_vector = [[float(x)], [float(y)], [float(z)]]
            color_coord = np.array([undist_col_points*float(translation_vector[2][0])])
            m2.fill_color_light_position(color_coord)
            lighting_case = 2
            color = (255,0,0)
        
        if lighting_case == 2 or lighting_case == 3:
                
            if medium=="water":
                translation_vector = np.asarray(translation_vector)*1.33  
      
            distance = np.linalg.norm(translation_vector)
            heading = np.degrees(np.arctan2(translation_vector[0][0],translation_vector[2][0]))
            elevation = np.degrees(np.arctan2(translation_vector[1][0],translation_vector[2][0]))
            
            if show_lights==True:
                for idx in range(contour_count):
                    image = cv2.circle(image, (int(dist_points[idx,0]), int(dist_points[idx,1])), 5, (0,255,0), -1) 
            	    #text = str(np.round_(light_coord[idx,0], decimals = 2))+","+str(np.round_(light_coord[idx,1], decimals = 2))
                    #cv2.putText(image,text,(int(p[0])-1, int(p[1])-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)
                for idx in range(col_contour_count):
                    image = cv2.circle(image, (int(col_points[idx,0]), int(col_points[idx,1])), 5, (255,0,0), -1)  
            

            if save_data_for_video==True:
                           
                text= "position in m:"+str(round(translation_vector[0][0],1))+","+str(round(translation_vector[1][0],1))+","+str(round(translation_vector[2][0],1))
                cv2.putText(image,text,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color,2,cv2.LINE_AA)
            
                text= "distance,heading,elevation in m,deg,deg:"+str(round(distance,2))+","+str(round(heading,1))+","+str(round(elevation,1))   
                cv2.putText(image,text,(20,image.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color,2,cv2.LINE_AA)
                cv2.imwrite("stream_video/"+str(stamp)+".jpg",image)

            m1.fill_target_heading_elevation(distance,heading,elevation)
            m1.fill_target_valid(1,1,1)
            m1.set_number_of_lights(contour_count, col_contour_count)
            m1.set_lighting_case(lighting_case)

            m2.set_number_of_lights(contour_count, col_contour_count)
            
        else:
            m1.fill_target_valid(0,0,0)
        """  
        m1.set_date_time()
        m1.set_checksum()
        m1.set_camera_status(success)
        m2.set_date_time()
        m2.set_checksum()
        m2.set_camera_status(success)

    
        cv2.imshow('frame',image)
        
        logger.info("Message 1 is %s",m1.convert_to_string())
        logger.info("Message 2 is %s",m2.convert_to_string())
    
        logger.info("Sending m1 to port:"+port1)
        server.sendto(m1.convert_to_string().encode(), ('<broadcast>', int(port1)))
        logger.info("Sending m2 to port:"+port2)
        server.sendto(m2.convert_to_string().encode(), ('<broadcast>', int(port2)))

    except Exception as ex:
        print(ex)
        print(sys.exc_info()[-1].tb_lineno)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, image = vidcap.read()      
    print('Image Count:', img_count)
    print('Detected: ',detect_count)
    if img_count:
        print('success rate ', (detect_count/img_count))
    print('Read a new frame: ', success)

