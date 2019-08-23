

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Reshape, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import ZeroPadding2D, Cropping2D, AveragePooling2D
from keras.layers import add, concatenate
from keras import backend as K


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn


def JH_Module(n_channel, tensor):

    x3 = Conv2D(filters=n_channel, kernel_size=(3, 3), strides=(1, 1), padding='same')(tensor)
    x3 = BatchNormalization(axis=-1)(x3)
    x3 = Activation('relu')(x3)

    x7 = Conv2D(filters=n_channel, kernel_size=(7, 7), strides=(1, 1), padding='same')(tensor)
    x7 = BatchNormalization(axis=-1)(x7)
    x7 = Activation('relu')(x7)

    x11 = Conv2D(filters=n_channel, kernel_size=(11, 11), strides=(1, 1), padding='same')(tensor)
    x11 = BatchNormalization(axis=-1)(x11)
    x11 = Activation('relu')(x11)

    out = concatenate([x11, x7, x3], axis=-1)

    return out


def JH_Module2(tensor):

    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(tensor)

    conv1_2 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)

    conv1_3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv1_3)

    MP1_4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(tensor)
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(MP1_4)

    out = concatenate([conv4, conv3, conv2, conv1], axis=-1)
    
    return out


def newNet_Juhui(InputShape, num_classes = 2, weights = None):		

    
    if num_classes == 2:
        num_classes = 1
        loss = 'binary_crossentropy'
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'



    data = Input(shape=InputShape, dtype = 'float', name = 'data')		# 224
    x = Lambda(mvn, name='mvn0')(data)

    x = JH_Module2(x)				
    #x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)			# 112
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = JH_Module2(x)	
    #x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)			# 56
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = JH_Module2(x)	
    #x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)			# 28
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = JH_Module2(x)								
    #x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)			# 14
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = JH_Module2(x)								
    # x = AveragePooling2D(pool_size=(2, 2))(x)


    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu', name='FC0')(x)
    x = Dense(100, activation='relu', name='FC1')(x)
    x = Dense(10, activation='relu', name='FC2')(x)

    predictions = Dense(num_classes, activation = activation, name = 'prediction')(x)


    model = Model(inputs=data, outputs=predictions, name = 'New_Juhui')


    if weights is not None:
        model.load_weights(weights)


    adam = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])


    return model

      

