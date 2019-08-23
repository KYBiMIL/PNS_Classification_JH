#!/usr/bin/env python2.7

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, GaussianDropout, Lambda, BatchNormalization
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense
from keras.layers import Concatenate, Add, Multiply
from keras.layers import Activation
from keras.layers import ZeroPadding2D, Cropping2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Permute
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




def CBAM_Channel_Attention(FM, reduction_ratio):

    if K.image_data_format() == "channel_first":
        ch_axis = 1
        #print("\n\nchannel first\n\n")
    else:
        ch_axis = -1	
        #print("\n\nchannel last\n\n")

    channel = FM._keras_shape[ch_axis]			# number of the channel

    # shared_MLP1 = Dense(channel//reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # shared_MLP2 = Dense(channel, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avr_map = GlobalAveragePooling2D()(FM)
    avr_map = Reshape((1, 1, channel))(avr_map)
    avr_map = Dense(channel//reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(avr_map)		# = shared_MLP1(avr_map)
    avr_map = Dense(channel, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(avr_map)						# = shared_MLP2(avr_map)


    max_map = GlobalMaxPooling2D()(FM)
    max_map = Reshape((1, 1, channel))(max_map)
    max_map = Dense(channel//reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(max_map)		# = shared_MLP1(max_map)
    max_map = Dense(channel, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(max_map)						# = shared_MLP2(max_map)

    # channel attention
    Mc = Add()([avr_map, max_map])
    Mc = Activation('sigmoid')(Mc)

    if K.image_data_format() == "channel_first":
        Mc = Parmute((3, 1, 2))(Mc)

    F_ = Multiply()([FM, Mc])

    return F_


def CBAM_Spatial_Attention(F_):

    k_size = 7											# kernel_size

    if K.image_data_format() == "channel_first":
        channel = F_._keras_shape[1]
        F_ = Permute((2, 3, 1))(F_)
    else:
        channel = F_._keras_shape[-1]
        F_ = F_

    avr_map = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(F_)    
    max_map = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(F_)

    concat = Concatenate(axis=3)([avr_map, max_map])
    # spatial attention
    Ms = Conv2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)		# sigmoid after convolution

    if K.image_data_format() == "channel_first":
        Ms = Permute((3, 1, 2))(Ms)

    F__ = Multiply()([F_, Ms])

    return F__




def BAM_Channel_Attention(FM, reduction_ratio):
        
    channel = FM._keras_shape[-1]        

    FM = GlobalAveragePooling2D()(FM)
    FM = Reshape((1, 1, channel))(FM)
    FM = Dense(channel//reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(FM)
    FM = Dense(channel, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(FM)
    
    Mc = BatchNormalization(axis=-1)(FM)

    return Mc


def BAM_Spatial_Attention(FM, reduction_retio):
    
    channel = FM._keras_shape[-1] 
    w = FM._keras_shape[-2]
    h = FM._keras_shape[-3]

    FM = Conv2D(filters=channel//reduction_retio, kernel_size=1, padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False)(FM)	
    FM = Conv2D(filters=channel//reduction_retio, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False)(FM)	
    FM = Conv2D(filters=channel//reduction_retio, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False)(FM)
    FM = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False)(FM)

    Ms = BatchNormalization(axis=-1)(FM)

    return Ms




def BAM_block(FM, reduction_ratio=8):

    #channel = FM._keras_shape[-1] 
    #w = FM._keras_shape[-2]
    #h = FM._keras_shape[-3]
    (_, h, w, channel) = FM._keras_shape

    Mc = BAM_Channel_Attention(FM, reduction_ratio)

    Mc = Lambda(lambda x: K.repeat_elements(x, h, axis=1))(Mc)
    print(Mc)
    Mc = Lambda(lambda x: K.repeat_elements(x, w, axis=2))(Mc)
    #Mc = K.repeat_elements(Mc, w, axis=2)
    print(Mc)

    Ms = BAM_Spatial_Attention(FM, reduction_ratio)
    
    Ms = Lambda(lambda x: K.repeat_elements(x, channel, axis=3))(Ms)
    #Ms = K.repeat_elements(Ms, channel, axis=3)
    print(Ms)

    M_F = Add()([Mc, Ms])
    print(M_F)
    M_F = Activation('sigmoid')(M_F)

    F_ = Multiply()([FM, M_F])
    F_ = Add()([F_, FM])

    return F_





def CBAM_block(FM, reduction_ratio=8):

    F_ = CBAM_Channel_Attention(FM, reduction_ratio)
    F__ = CBAM_Spatial_Attention(F_)

    return F__




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

def identity_block_with_CBAM(x, f, channel, stage, block):        # skips over 2 layers

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

    F__ = CBAM_block(x)
    out = Add()([F__, x_shortcut])

    out = Activation('relu')(out)

    return out

def identity_block_with_BAM(x, f, channel, stage, block):        # skips over 2 layers

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

    F_ = BAM_block(x)
    out = Add()([F_, x_shortcut])

    out = Activation('relu')(out)

    return out


def JuhuiModelResBAM(input_shape, num_classes, weights=None):
 
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
    # DO4 = Dropout(rate=0.5)(C4b)/data/PNS/PNS_Classification_juhui2/data/ROI_Rotation_AHE
    P4 =  MaxPooling2D(pool_size=(2, 2))(C4b)						# 25

    C5a = identity_block(P3, f, 1024, stage=4, block='a')
    C5b = identity_block(C4a, f, 1024, stage=4, block='b')

    """


    x = identity_block_with_BAM(x, f, 64, stage=1, block='a')		# 224
    x = identity_block_with_BAM(x, f, 64, stage=1, block='b')
    x = MaxPooling2D(pool_size=(2, 2))(x)                    # 112

    x = identity_block_with_BAM(x, f, 128, stage=2, block='a')	
    x = identity_block_with_BAM(x, f, 128, stage=2, block='b')
    x = MaxPooling2D(pool_size=(2, 2))(x)                    # 56

    x = identity_block_with_BAM(x, f, 256, stage=3, block='a')
    x = identity_block_with_BAM(x, f, 256, stage=3, block='b')
    x = Dropout(rate=0.5)(x)
    # DO3 = Dropout(rate=0.5)(C3b)
    x =  MaxPooling2D(pool_size=(2, 2))(x)                   # 50

    x = identity_block_with_BAM(x, f, 512, stage=4, block='a')
    x = identity_block_with_BAM(x, f, 512, stage=4, block='b')
    x = Dropout(rate=0.5)(x)
    # DO4 = Dropout(rate=0.5)(C4b)
    x =  MaxPooling2D(pool_size=(2, 2))(x)						# 14

    x = identity_block_with_BAM(x, f, 1024, stage=5, block='a')
    x = identity_block_with_BAM(x, f, 1024, stage=5, block='b')

    print("\ntest\n")
    print(x)

    x = GlobalAveragePooling2D()(x)
    print(x)
    x = Dense(512, activation = 'relu', name='FC0')(x)
    print(x)
    x = Dense(256, activation = 'relu', name='FC1')(x)
    x = Dense(100, activation = 'relu', name='FC2')(x)
    x = Dense(10, activation = 'relu', name='FC3')(x)
    print(x) 

    predictions = Dense(num_classes, activation = activation, name = 'prediction')(x)
    print(predictions)

    model = Model(inputs=data, outputs=predictions, name = 'ResModelJuhui')


    if weights is not None:
        model.load_weights(weights)

    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.5, nesterov=True)
    rmsp = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=1e-5)
    model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = JuhuiModelRes((224, 224, 3), 2, weights=None)



