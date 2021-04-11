###############################################################################
## getModelSummary.py
## Author: Jason Epp
## Description: This script just loads a model from JSON file and prints the
## model summary
##############################################################################

import keras
from keras import layers
from keras.models import Sequential, Model, model_from_json

import pickle
import argparse

arg_p = argparse.ArgumentParser()
arg_p.add_argument('-c', '--config', type=str, required=True, 
            help='usage: "-c <configuration file path>"')

args = vars(arg_p.parse_args())

cfg_path = args["config"]

with open(cfg_path, 'rb') as f_cfg:
    cfg = pickle.load(f_cfg)
    model = model_from_json(cfg)

model.summary()
for m_layer in model.layers:
    try:
        m_layer.summary()
    except:
        pass