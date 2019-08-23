#!/usr/bin/env python2.7

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, GaussianDropout, Lambda, BatchNormalization
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense
from keras.layers import concatenate, Add
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



def JuhuiModelRes(input_shape, num_classes, weights=None):
 
    if num_classes == 2:
        num_classes = 1
        loss = 'binary_crossentropy'
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'



    kwargs2 = dict(
        kernel_size=3,
        strides=2,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )

    f = 3
    print("Input_size = ", input_shape)
    data = Input(shape=input_shape, dtype='float', name='data')  # 400x400x3
    x = Lambda(mvn, name='mvn0')(data)
    """
    C1a = identity_block(data, f, 64, stage=1, block='a')	
    C1b = identity_block(C1a, f, 64, stage=1, block='b')
    P1 = MaxPooling2D(pool_size=(2, 2))(C1b)                    # 200

    C2a = identity_block(P1, f, 128, stage=2, block='a')	
    C2b = identity_block(C2a, f, 128, stage=2, block='b')
    P2 = MaxPooling2D(pool_size=(2, 2))(C2b)                    # 100

    C3a = identity_block(P2, f, 256, stage=3, block='a')
    C3b = identity_block(C3a, f, 256, stage=3, block='b')
    # DO3 = Dropout(rate=0.5)(C3b)
    # P3 =  MaxPooling2D(pool_size=(2, 2))(DO3)                   # 15
    P3 =  MaxPooling2D(pool_size=(2, 2))(C3b)                   # 50

    C4a = identity_block(P3, f, 512, stage=4, block='a')
    C4b = identity_block(C4a, f, 512, stage=4, block='b')
    # DO4 = Dropout(rate=0.5)(C4b)
    P4 =  MaxPooling2D(pool_size=(2, 2))(C4b)						# 25

    C5a = identity_block(P3, f, 1024, stage=4, block='a')
    C5b = identity_block(C4a, f, 1024, stage=4, block='b')

    """


    x = identity_block(x, f, 64, stage=1, block='a')		# 224
    #x = identity_block(x, f, 64, stage=1, block='b')
    x = MaxPooling2D(pool_size=(2, 2))(x)                    # 112

    x = identity_block(x, f, 128, stage=2, block='a')	
    #x = identity_block(x, f, 128, stage=2, block='b')
    x = MaxPooling2D(pool_size=(2, 2))(x)                    # 56

    x = identity_block(x, f, 256, stage=3, block='a')
    #x = identity_block(x, f, 256, stage=3, block='b')
    # DO3 = Dropout(rate=0.5)(C3b)
    # P3 =  MaxPooling2D(pool_size=(2, 2))(DO3)                   # 28
    x =  MaxPooling2D(pool_size=(2, 2))(x)                   # 50

    x = identity_block(x, f, 512, stage=4, block='a')
    #x = identity_block(x, f, 512, stage=4, block='b')
    # DO4 = Dropout(rate=0.5)(C4b)
    x =  MaxPooling2D(pool_size=(2, 2))(x)						# 14

    #x = identity_block(x, f, 1024, stage=5, block='a')
    #x = identity_block(x, f, 1024, stage=5, block='b')


    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation = 'relu', name='FC0')(x)
    x = Dense(256, activation = 'relu', name='FC1')(x)
    x = Dense(100, activation = 'relu', name='FC2')(x)
    x = Dense(10, activation = 'relu', name='FC3')(x)
    predictions = Dense(num_classes, activation = activation, name = 'prediction')(x)


    model = Model(inputs=data, outputs=predictions, name = 'ResModelJuhui')


    if weights is not None:
        model.load_weights(weights)

    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = JuhuiModelRes((224, 224, 3), 2, weights=None)



