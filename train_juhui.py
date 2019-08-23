import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import random
import h5py
import json
import argparse
import datetime
from pandas_ml import ConfusionMatrix
from datetime import datetime as T
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from configparser import ConfigParser
from keras import optimizers

#from Model_juhui import AlexNet_Juhui
from ResNet_juhui import JuhuiModelRes
#from ResNet_juhui2 import JuhuiModelRes
from newNet_juhui import newNet_Juhui

import matplotlib
import matplotlib.pylab as pltap

from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.models import load_model
from keras.models import Input
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.utils import np_utils
from keras.utils.training_utils import multi_gpu_model





random_seed = 1234
np.random.seed(random_seed)

n_batches = 128
n_epoches = 1000

learning_rate = 0e-6

ROWS, COLS      = 224, 224


## Setting GPU Environment

parser = argparse.ArgumentParser()
parser.add_argument("GID",
                     help="ID of GPU { 0 | 1 | 2 | 3 }. Default = 0,1,2,3",
                     nargs='?',
                     const=1,
                     default="0,1,2,3")
args      = parser.parse_args()

GPUID     = args.GID #sys.argv[1]
#len(GPUID) >= 1:

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # change available GPU #




## Route/Path Determination

CWD             = os.path.dirname(os.path.abspath(__file__))
#CWD = 'F:/expPNS5/fold2'
FOLDID          = CWD.split("/")[-1]
print(FOLDID)
EXPCODE         = CWD.split("/")[-2]
print(EXPCODE)
print(CWD)

LOGDIR          = os.path.join(CWD, "log")
LOGNAME         = "log_%s_%s.txt"%(EXPCODE, FOLDID)
LOGFILE         = os.path.join(LOGDIR, LOGNAME)
CSVNAME         = "csv_%s_%s.txt"%(EXPCODE, FOLDID)
CSVFILE         = os.path.join(LOGDIR, CSVNAME)
MODELDIR        = os.path.join(CWD,"model")
MODELNAME       = "%s/mdl_%s_%s.h5"%(MODELDIR, EXPCODE, FOLDID)
OUTPUTNAME      = "out_%s_%s.txt"%(EXPCODE, FOLDID)
DATADIR         = os.path.join(CWD, "data")

n_classes       = 2





def set_callback_list(exp_code):

  model_file = "TestServer"

  ckpt_filepath = '../model/mdl_' + exp_code + model_file + '_{epoch:04d}.h5'
  checkpoint = ModelCheckpoint(ckpt_filepath,
                               monitor='val_acc',
                               verbose=2,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='max')

  tensorboard = TensorBoard(log_dir='./log/' + model_file,
                              histogram_freq=0,
                              batch_size=n_batches,
                              write_graph=True,
                              write_images=True)

  csv_filename = '../log/log_' + exp_code + model_file + '.log'
  csv_logger   = CSVLogger(csv_filename, separator='\t', append=False)

  return [checkpoint, tensorboard, csv_logger]





if __name__ == '__main__':

  ST = T.now()				# call date and time

  train_path       = "/data/PNS_CLASSIFICATION/PNS_DATA/TestServerData/train"
  train_datagen    = ImageDataGenerator(
						# preprocessing_function=preprocess_input,
						rotation_range=270,
						shear_range=0.5,
						horizontal_flip=True)
  train_generator  = train_datagen.flow_from_directory(train_path,
                                        target_size=(ROWS, COLS),
                                        batch_size=n_batches,
                                        class_mode='binary',
                                        shuffle=True)


  test_path        = "/data/PNS_CLASSIFICATION/PNS_DATA/TestServerData/test"
  test_datagen     = ImageDataGenerator()
  test_generator   = test_datagen.flow_from_directory(test_path,
                                       target_size=(ROWS, COLS),
                                       batch_size=n_batches,
                                       class_mode='binary',
                                       shuffle=False)

  # print("len(test_gen) = ", len(test_generator))
  Input_shape = (ROWS, COLS, 3)
  # model = AlexNet_Juhui(InputShape = Input_shape, n_class = 2, weights = None)
  # model = ResNet_Juhui(InputShape = Input_shape, n_class = n_classes, weights = None)
  # model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=Input_shape, pooling=None, classes=2)

  # model = JuhuiModelRes(input_shape = Input_shape, num_classes = n_classes, weights = None)
  model = JuhuiModelRes(Input_shape, n_classes, weights = None)

  # optimization
  # sgd = optimizers.SGD(lr = 0.01, momentum = 0.9, nesterov = True)
  # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


  prefix = "%s_%s"%(EXPCODE,FOLDID)
  hist = model.fit_generator(train_generator,
                             steps_per_epoch=len(train_generator)/n_batches,
                             epochs=n_epoches,
                             verbose=2,
                             callbacks=set_callback_list(prefix),
                             validation_data=test_generator,
                             validation_steps=len(test_generator))

  ET = T.now()
  print("Elapsed Time =", ET-ST)


