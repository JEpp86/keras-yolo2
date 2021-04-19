# YOLOv2 in Keras and Applications

This repo contains the implementation of YOLOv2 in Keras with Tensorflow backend. It supports training YOLOv2 network with various backends such as MobileNet and InceptionV3, as well as Very Tiny YOLO model developed for embedded appications. This repo was used as a basis for Deep Learning Systems in Engineering project

## Usage for python code

### 0. Requirement

python 3.5

keras >= 2.0.8

tensorflow 1.x

imgaug

### 1. Data preparation
dataset can be found here:

Organize the dataset into 2 folders:

+ ../data/training <= the folder that contains the train images.
+ ../data/annotation_voc <= the folder that contains the train annotations in VOC format.

relative to the project root dierctory, or modify the configuration file based of instuctions below

There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically split into the training set and validation set using the ratio of 0.8.

### 2. Edit the configuration file
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "architecture":         "Full Yolo",    # "Tiny Yolo" or "Full Yolo" or "MobileNet" or "SqueezeNet" or "Inception3"
        "input_size":           416,
        "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        "max_box_per_image":    10,        
        "labels":               ["raccoon"]
    },

    "train": {
        "train_image_folder":   "/home/andy/data/raccoon_dataset/images/",
        "train_annot_folder":   "/home/andy/data/raccoon_dataset/anns/",      
          
        "train_times":          10,             # the number of time to cycle through the training set, useful for small datasets
        "pretrained_weights":   "",             # specify the path of the pretrained weights, but it's fine to start from scratch
        "batch_size":           16,             # the number of images to read in each batch
        "learning_rate":        1e-4,           # the base learning rate of the default Adam rate scheduler
        "nb_epoch":             50,             # number of epoches
        "warmup_epochs":        3,              # the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically

        "object_scale":         5.0 ,           # determine how much to penalize wrong prediction of confidence of object predictors
        "no_object_scale":      1.0,            # determine how much to penalize wrong prediction of confidence of non-object predictors
        "coord_scale":          1.0,            # determine how much to penalize wrong position and size predictions (x, y, w, h)
        "class_scale":          1.0,            # determine how much to penalize wrong class prediction

        "debug":                true            # turn on/off the line that prints current confidence, position, size, class losses and recall
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",

        "valid_times":          1
    }
}

```

The model section defines the type of the model to construct as well as other parameters of the model such as the input image size and the list of anchors. The ```labels``` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network. The rest images are simply ignored.

### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

### 4. Pretrain the model
unsure the data from the data zip is on the followig location:
https://drive.google.com/file/d/1adfK0OyuQiaBN9gPpGSt9CeiuoQqTppC/view?usp=sharing
../data/Dock 
../data/NotDock

to run pretraining use
'python PretrainBackendModels.py [-option]'

to pretrain the very tiny YOLO model run 
'python PretrainBackendModels.py -ts

Note: if a backend modek isn't pretrained before attempting training the model will fail at creating in the train script.


### 5. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 6. Perform detection using trained weights on an image by running
Using demo code matching project (using json model cfg, and hd5 weights file)
'sh run_demo.sh' for final model and wights
from python directly:
python docking_demo.py -cfg <model.cfg> -w <weights.hd5> -src video -video dockvideo.avi 
for testing various models

This carries out detection while also measuring statistics. This will save images to stream_training/ directory in project root folder

As provided in previous repo (.h5 file and training config) 
`python predict.py -c config.json -w /path/to/best_weights.h5 -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.


## Copyright

See [LICENSE](LICENSE) for details.
