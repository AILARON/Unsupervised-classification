##################
#Data augmentation class used to artificially  expand
#the size of the training dataset by creating modified versions
#of images in the existing dataset.

# This file is based on the tensorflow keras preprocessing library
#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing

"""
Input: 1. Path to folders with jpg pictures. ex: '/directory'
Input: 2. Path to savefolder ex: '/directory/augmented_data'
Output: os folder with folders for each class found in '/directory'
"""
##################

from tensorflow.keras.preprocessing.image import ImageDataGenerator,save_img
import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm


class DataAugmentation:
    load_data_dir = None
    save_data_dir = None

    def __init__(self,load_data_dir,save_data_dir):
        self.load_data_dir = load_data_dir
        self.save_data_dir = save_data_dir

    def dataAugmentation(self,):
        datagen = ImageDataGenerator(
            rescale = 1. / 255,\
            zoom_range=0.1,\
            rotation_range=90,\
            width_shift_range=0.1,\
            height_shift_range=0.1,\
            horizontal_flip=True,\
            vertical_flip=True)

        generator = datagen.flow_from_directory(
            directory=self.load_data_dir,
            target_size=(64, 64),
            color_mode="grayscale",
            batch_size=1000,
            class_mode="categorical",
            shuffle=False)

        #Get information of loaded images
        batch_size = 100
        num_samples = generator.n
        num_classes = generator.num_classes
        input_shape = generator.image_shape
        class_names = [k for k,v in generator.class_indices.items()]
        print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))

        #Make directories to save the images
        os.makedirs(os.path.join(self.save_data_dir), exist_ok=True)
        for label in class_names:
            os.makedirs(os.path.join(self.save_data_dir, label), exist_ok=True)

        # Iterate over images and save in class folders
        x, y = generator.next()
        for i in range(80):
            x, y = generator.next()
            j = 0
            for x, y in tqdm(zip(x, y)):
                filename = os.path.join(self.save_data_dir, class_names[y.argmax()], "_"+str(i*batch_size+j)+"_"+ '.jpg')
                tf.keras.preprocessing.image.save_img(
                filename, x, data_format=None, file_format=None, scale=True
                )
                j = j+1
