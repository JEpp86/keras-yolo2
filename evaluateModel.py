#Evaluate Model

import json
import pickle
import argparse
import os
import numpy as np
from frontend import YOLO
from preprocessing import parse_annotation

import tensorflow as tf
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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Evaluate YOLO_v2 model against validation dataset')

argparser.add_argument('-c', '--conf', required=True, help='path to configuration file')
argparser.add_argument('-m', '--model', required=True, help='path to json model (cfg)')
argparser.add_argument('-w', '--weights', required=True, help='path to weights (hd5)')
args = argparser.parse_args()

config_path = args.conf
model_path = args.model
weights_path = args.weights

with open(config_path) as config_buffer:    
    config = json.loads(config_buffer.read())

yolo = YOLO(backend             = None,
            input_size          = config['model']['input_size'], 
            labels              = config['model']['labels'], 
            max_box_per_image   = config['model']['max_box_per_image'],
            anchors             = config['model']['anchors'], 
            load_from_json      = model_path,
            trained_weights     = weights_path)
yolo.set_batch(config['train']['batch_size'])

#TODO insert bactch generator here
if os.path.exists(config['valid']['valid_annot_folder']):
    valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                config['valid']['valid_image_folder'], 
                                                config['model']['labels'])
else:
    valid_imgs, valid_labels = parse_annotation(config['train']['train_annot_folder'], 
                                            config['train']['train_image_folder'], 
                                            config['model']['labels'])

valid_generator = yolo.get_generator_from_data(valid_imgs)

average_precisions, average_speed = yolo.evaluate(valid_generator)


# print evaluation
for label, average_precision in average_precisions.items():
    print(yolo.get_label(label), '{:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
print('speed: {:.4f}'.format(average_speed))

