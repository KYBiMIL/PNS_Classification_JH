#!/usr/bin/env python2.7

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, GaussianDropout, Lambda, BatchNormalization
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense, AveragePooling2D
from keras.layers import concatenate, Add, Multiply
from keras.layers import Activation
from keras.layers import ZeroPadding2D, Cropping2D, GlobalAveragePooling2D
from keras import backend as K
from keras.initializers import glorot_uniform

def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn

def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h / 2, crop_h / 2 + rem_h)
    crop_w_dims = (crop_w / 2, crop_w / 2 + rem_w)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])

    return cropped





def identity_block(x, f, channel, stage, block):        # skips over 2 layers

    conv_namebase = 'res' + str(stage) + block + '_branch'
    bn_namebase = 'bn' + str(stage) + block + '_branch'
    x_shortcut = x

    x = Conv2D(filters=channel, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_namebase+'2a', kernel_initializer=glorot_uniform(seed=1234))(x)
    x = BatchNormalization(axis=-1, name=bn_namebase+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=channel, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_namebase+'2b', kernel_initializer=glorot_uniform(seed=1234))(x)
    x = BatchNormalization(axis=-1, name=bn_namebase+'2b')(x)

    x_shortcut = Conv2D(filters=channel, kernel_size=(1, 1), strides=(1, 1), name=conv_namebase+'1', kernel_initializer=glorot_uniform(seed=1234))(x_shortcut)
    x_shortcut = BatchNormalization(axis=-1, name=bn_namebase+'1')(x_shortcut)

    x = Add()([x, x_shortcut])

    x = Activation('relu')(x)

    return x

def Model0(x, num_classes):

    if num_classes == 2:
        num_classes = 1
        loss = 'binary_crossentropy'
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'


    f = 3

    ## for Model0, smaller input
    data0 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x0 = identity_block(data0, f, 64, stage=1, block='a0')		# 44
    x0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x0)                    

    x0 = identity_block(x0, f, 128, stage=2, block='a0')		# 22
    x0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x0)                    

    x0 = identity_block(x0, f, 256, stage=3, block='a0')		# 11
    # x0 =  MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x0)                   

    # x0 = identity_block(x0, f, 512, stage=4, block='a0')		# 14

    #g0 = GlobalAveragePooling2D()(x0)
    # predictions0 = Dense(num_classes, activation = activation, name = 'prediction0')(g0)

    return x0



def Model1(x, num_classes):

    if num_classes == 2:
        num_classes = 1
        loss = 'binary_crossentropy'
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'


    f = 3

    ## for Model1, larger input     
     
    x1 = identity_block(x, f, 64, stage=1, block='a1')		# 88
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)                    

    x1 = identity_block(x1, f, 128, stage=2, block='a1')		# 44
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)                    

    x1 = identity_block(x1, f, 256, stage=3, block='a1')		# 22
    x1 =  MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)                   

    x1 = identity_block(x1, f, 512, stage=4, block='a1')		# 11
    # x1 =  MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)						

    # x1 = identity_block(x1, f, 1024, stage=4, block='b')		# 14

    #g1 = GlobalAveragePooling2D()(x1)
    # predictions1 = Dense(num_classes, activation = activation, name = 'prediction1')(g1)

    return x1


def MultiRes_JuhuiModel(input_shape_1, num_classes, weights=None):
 
## input_shape_1 is larger than input_shape_0

    if num_classes == 2:
        num_classes = 1
        loss = 'binary_crossentropy'
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'

    data = Input(shape=input_shape_1, dtype='float', name='data')
    x = Lambda(mvn, name='mvn')(data)

    x0 = Model0(x, num_classes)
    x1 = Model1(x, num_classes)    


    ## for combined model
    xC = concatenate([x0, x1])
    xC = GlobalAveragePooling2D()(xC)
    xC = Dense(256, activation='relu', name='FC0')(xC)
    xC = Dense(96, activation='relu', name='FC1')(xC)

    predictionC = Dense(num_classes, activation=activation, name='combined_predictions')(xC)

    modelC = Model(inputs=data, outputs=predictionC)

    sgdC = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.7, nesterov=True)
    modelC.compile(optimizer=sgdC, loss=loss, metrics=['accuracy'])

    print("Model1 is Possible!!!!!!!!!!!!!!!!!!!!!!!!!?")
    return modelC
   


if __name__ == '__main__':
    model = MultiRes_JuhuiModel((224, 224, 3), 2, weights=None)



