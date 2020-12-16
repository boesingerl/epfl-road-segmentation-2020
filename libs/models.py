import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

def last_layers(input_tensor):
    x = Conv2D(24, 5, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, strides=1, padding='same', )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(8, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(4, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(2, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    return x

def unet(input_size = (256,256,3), post_processing=True,batch_size=16, levels=5, activation='relu', kernel_initializer='he_normal', 
         dropout_p=0.2, dropout_max_level_only=True, bn_pos1 = True, bn_pos2 = True, optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'], pretrained_weights = None):
    
    inputs = keras.Input(input_size, batch_size,name='input')
    o = inputs
    levels = [2**(6+i) for i in range(levels)]
    c_layers = {}
    
    for l in levels:
        o = layers.Conv2D(l, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='conv1down'+str(l))(o)
        o = layers.BatchNormalization(name='batch1down'+str(l))(o) if bn_pos1 else o
        o = layers.Conv2D(l, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='conv2down'+str(l))(o)
        o = layers.BatchNormalization(name='batch2down'+str(l))(o) if bn_pos2 else o
        o = layers.Dropout(dropout_p,name='dropout'+str(l))(o) if dropout_p > 0 and dropout_max_level_only != True else o
        c_layers[l] = o
        o = layers.MaxPooling2D(pool_size=(2, 2),name='pooling'+str(l))(o) if l != max(levels) else layers.Dropout(dropout_p)(o) if dropout_p > 0 else o
    
    for l in reversed(levels[:-1]):
        o = layers.Conv2D(l, 2, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='conv1up'+str(l))(layers.UpSampling2D(size = (2,2))(o))
        o = layers.concatenate([c_layers[l],o],name='concat'+str(l), axis = 3)
        o = layers.Conv2D(l, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='conv2up'+str(l))(o)
        o = layers.BatchNormalization(name='batch1up'+str(l))(o) if bn_pos1 else o
        
        o = layers.Conv2D(l, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='conv3up'+str(l))(o)
        o = layers.BatchNormalization(name='batch2up'+str(l))(o) if bn_pos2 else o

        if l == min(levels):
            o = layers.Conv2D(2, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='lastconv')(o)
            o = layers.Conv2D(1, 1, activation = 'sigmoid',name='lastconv2')(o)
    if post_processing:
      o = last_layers(o)
    model = keras.Model(inputs, o)

    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights,by_name=True)

    return model