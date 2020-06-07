#####################
#Load dataset class based on tensorflow implementation
#https://www.tensorflow.org/tutorials/load_data/images
"""
Input: Path to folders with jpg pictures. ex: '/directory'
Output: Image dataset train, train_label, test, test_label
"""
####################

import tensorflow as tf
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
AUTOTUNE=tf.data.experimental.AUTOTUNE


class LoadDataset:
    data_dir = None
    list_ds = None

    IMAGE_COUNT = None
    CLASS_NAMES = None
    TEST_SIZE = 0.1

    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_DEPTH = 1


    def __init__(self,data_dir,test_size):
        self.data_dir = pathlib.Path(data_dir)

        self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.jpg')))
        print("found ", self.IMAGE_COUNT," images in folder")

        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "test"])
        print("found ",self.CLASS_NAMES," classes")

        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))

        self.TEST_SIZE = test_size

    def get_label(self,file_path):
        parts = tf.strings.split(file_path, '/')
        return parts[-2] == self.CLASS_NAMES

    def decode_img(self,img):
        img = tf.image.decode_jpeg(img, channels=1) #color images
        img = tf.image.convert_image_dtype(img, tf.float32)
        #convert unit8 tensor to floats in the [0,1]range
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self,file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def load_data(self):
        labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        ds = np.zeros((self.IMAGE_COUNT,self.IMG_HEIGHT,self.IMG_WIDTH,self.IMG_DEPTH))
        ds_label = np.zeros(self.IMAGE_COUNT)

        i = 0
        for image, label in (labeled_ds):
          ds[i] = image
          ds_label[i]= np.where(label.numpy())[0]
          i  = i + 1

        if self.TEST_SIZE > 0:
            train, test, train_label, test_label = train_test_split(ds, ds_label, test_size=self.TEST_SIZE)
            print ("training data shape",train.shape, train_label.shape)
            print ("testing data shape",test.shape, test_label.shape)
            return train, train_label, test, test_label
        else:
            train = ds
            train_label = ds_label
            test = 0
            test_label = 0
            print ("training data shape",train.shape, train_label.shape)
            
            return train, train_label, test, test_label

    def get_list_ds(self):
        return self.list_ds
