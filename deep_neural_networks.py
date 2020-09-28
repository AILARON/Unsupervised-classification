



import tensorflow as tf





#RESNET
def RESNET101(input_shape = (64,64,3),output_shape = 121):
    #model= tf.keras.applications.ResNet101V2(include_top=True,weights=None, classes = 7)
    model= tf.keras.applications.ResNet101V2(include_top=False,weights=None)
    #add top layer
    layer =tf.keras.layers.GlobalAveragePooling2D()(model.output)
    layer = tf.keras.layers.Dense(output_shape, activation='softmax')(layer)
    model = tf.keras.models.Model(inputs=model.input,
                       outputs=layer, name="RESNET101")
    return model

#VGG16
def VGG(input_shape=(224,224,3),output_shape = 121):
    model= tf.keras.applications.VGG16(include_top=False,weights=None, input_shape=(224,224,3),
    )
    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(121, activation='softmax')(x)

    vgg = tf.keras.models.Model(inputs, x,name="VGG16")
    return model

def VGG_BATCHNORM(input_shape=(64,64,3),output_shape = 121):
    """
    Convolutional auto-encoder model, symmetric.
    Using the cnn network implementation VGG16 as feature extractor
    """
    # define the input to the encoder
    #inputShape = input_shape
    inputs = tf.keras.layers.Input(shape=input_shape)

    # apply 2 x (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same" )(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)


    # apply 2 x (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)


    # apply 3 x (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)


    # apply 3 x (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    #x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

    vgg =  tf.keras.models.Model(inputs, x,name="VGG16")

    return vgg



def COAPNET(input_shape = (64,64,3), output_shape = 121):
    """
    Convolutional auto-encoder model, symmetric.
    Using the cnn network implementation COAPNET as feature extractor
    """
    # define the input to the encoder
    #inputShape = (64, 64, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)

    # apply (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same" )(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)


    # apply (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)


    # apply (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)


    # apply (CONV => BN layer => ReLU activation) + MaxPooling
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same" )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

    return tf.keras.models.Model(inputs, x,name="COAPNET")


#based on https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
#Currently not used!
def RESNET():
    num_res_net_blocks = 20
    # define the input to the encoder
    inputShape = (64, 64, 3)
    inputs = tf.keras.layers.Input(shape=inputShape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Dense(256, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(121, activation='softmax')(x)
    return tf.keras.models.Model(inputs, outputs, name="RESNET")



def res_net_block(input_data, filters, conv_size):
  x = tf.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Add()([x, input_data])
  x = tf.keras.layers.Activation('relu')(x)
  return x







































"""
#based on https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691

import tensorflow as tf
from sklearn.cluster import KMeans

import tensorflow as tf
import numpy as np
from cbof import BoF_Pooling
from regularizer import SparseActivityRegularizer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


import tensorflow as tf

def res_identity(x, filters):
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block
  f1, f2 = filters

  #first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=("l2"))(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=("l2"))(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=("l2"))(x)
  x = BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input
  x = Add()([x, x_skip])
  x = Activation("relu")(x)

  return x


def res_conv(x, s, filters):
  '''
  here the input size changes'''
  x_skip = x
  f1, f2 = filters

  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=("l2"))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=("l2"))(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=("l2"))(x)
  x = BatchNormalization()(x)

  # shortcut
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',kernel_regularizer=("l2"))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add
  x = Add()([x, x_skip])
  x = Activation("relu")(x)

  return x


def resnet50():

  input_im = Input(shape=(64, 64, 1)) # cifar 10 images size
  x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = AveragePooling2D((2, 2), padding='same')(x)

  x = Flatten()(x)
  x = Dense(121, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

  # define the model

  model = Model(inputs=input_im, outputs=x, name='Resnet50')



  return model
"""
