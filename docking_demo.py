#!/usr/bin/env python


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




ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weight", type=str, required=True,help="weights")
ap.add_argument("-cfg", "--config", type=str, required=True,help="configuration")
ap.add_argument("-src", "--source", type=str, required=False,choices=['video','stream'],help="video or stream")
ap.add_argument("-l", "--lights", type=str, required=False,choices=['True','False'],help="show light coordinates or not")
ap.add_argument("-video", "--video", type=str, required=False,help="video for processing")
ap.add_argument("-slink", "--stream_link", type=str, required=False,help="stream link")


args = vars(ap.parse_args())
weights_path = args["weight"]
config_path = args["config"]
source =  "stream" if args["source"]==None else args["source"]
show_lights = True if args["lights"]=='True' else False
video_link =  "stream2.mp4" if args["video"]==None else args["video"]
default_stream_link = 'rtsp://admin:admin@192.168.214.40/h264.sdp?res=half&x0=0&y0=0&x1=1920&y1=1080&qp=16&doublescan=0&ssn=41645' 
stream_link = default_stream_link if args["stream_link"]==None else args["stream_link"]


print("Weights path is ",weights_path)
print("Config path is ",config_path)
print("Source is",source)
print("show lights is",show_lights)
print(args)


LABELS = ['Dock']
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
#ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
#ANCHORS          = [0.1,0.1,0.7,0.7,2,2,5,5,9,9]
ANCHORS          = [1.05, 1.68, 1.35, 2.06, 1.50, 2.40, 1.85, 2.82] #anchors determined by k.means clustering
TRUE_BOX_BUFFER  = 10


if source=="stream":
        vidcap = cv2.VideoCapture(stream_link)
else:
        vidcap = cv2.VideoCapture(video_link)


with open(config_path,'rb') as f:
    cfg = pickle.load(f)
    model = model_from_json(cfg)
    

with open(weights_path,'rb') as f:
    weights = pickle.load(f)
    model.set_weights(weights)

success,image = vidcap.read()
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
    try:
        frameId = int(round(vidcap.get(1)))
        img_count += 1

        stamp = str(time.time()).split('.')
        stamp = stamp[0] + stamp[1]       
        while( len(stamp)<17):
            stamp += '0' 
        input_image = cv2.resize(image, (224, 224))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)

        netout = model.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0],obj_threshold=0.3,nms_threshold=0.05,anchors=ANCHORS,nb_class=CLASS)            
        
        max_score = -1
        saved_box = None

        for bbox in boxes:
                if bbox.get_score()>max_score:
                        saved_box = bbox
        image_h, image_w, _ = image.shape  
 
        if saved_box == None:
                logger.info('Target not detected')
                print('No Target')        
                cv2.imwrite("stream_training/"+ str(stamp)+".jpg",image)
        else:

                for box in boxes:
                    print('-------------------')
                    print(box.get_score())
                    print(box.xmin, box.ymin)
                    print(box.xmax, box.ymax)                    
                image = draw_boxes(image, boxes, LABELS)
                detect_count += 1
                cv2.imwrite("stream_training/"+ str(stamp)+".jpg",image)
        
        cv2.imshow('frame',image)
        

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

