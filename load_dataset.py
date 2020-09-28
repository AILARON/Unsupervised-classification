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
import sys

from sklearn.model_selection import train_test_split
AUTOTUNE=tf.data.experimental.AUTOTUNE


KAGGLE_ORIGINAL_TRAIN = "dataset/kaggletrainoriginalfull/"
KAGGLE_ORIGINAL_TEST = "dataset/kaggle_original_train/"
WHOI_ORIGINAL_TRAIN_2014 = "dataset/WHOI/2014/"
WHOI_ORIGINAL_TRAIN_2013 = "dataset/WHOI/2013/"
LUOTRAINING = "dataset/Luo_Training/"
PASTORETRAINING ="dataset/Pastore_Training"
AILARONTRAINING = "dataset/Ailaron_Training"
AILARONTEST = "dataset/Ailaron_Test"

DATA_FORMAT = ["jpg","png"]

def importWHOI():
    loader = LoadDataset(WHOI_ORIGINAL_TRAIN_2014,0,True)
    #train_data, train_label, val, val_label = loader.load_data()

    return loader.load_data()

def importKaggleTrain(depth = 1):
    if depth == 3:
        loader = LoadDataset(KAGGLE_ORIGINAL_TRAIN,0,False,depth = 3)
        return loader.load_data()
    else:
        loader = LoadDataset(KAGGLE_ORIGINAL_TRAIN,0,False)
        return loader.load_data()

def importKaggleTest(depth = 1):
    if depth == 3:
        loader = LoadDataset(KAGGLE_ORIGINAL_TEST,0,False,depth = 3)
        return loader.load_data()
    else:
        loader = LoadDataset(KAGGLE_ORIGINAL_TEST,0,False)
        return loader.load_data()

def importLuoTraining(depth=1):
    if depth == 3:
        loader = LoadDataset(LUOTRAINING,0,False,depth = 3)
        return loader.load_data()
    else:
        loader = LoadDataset(LUOTRAINING,0,False)
        return loader.load_data()

def importPastoreTraining(depth=1):
    if depth == 3:
        loader = LoadDataset(PASTORETRAINING,0,False,depth = 3)
        return loader.load_data()
    else:
        loader = LoadDataset(PASTORETRAINING,0,False)
        return loader.load_data()

def importAilaronTraining(depth=1):
    loader = LoadDatasetTIFF(AILARONTRAINING,0,False)
    return loader.load_data(AILARONTRAINING)

def importAilaronTest(depth=1):
    loader = LoadDatasetTIFF(AILARONTEST,0,False)
    return loader.load_data(AILARONTEST)

class LoadDataset:
    data_dir = None
    list_ds = None

    IMAGE_COUNT = None
    CLASS_NAMES = None
    TEST_SIZE = 0.1

    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_DEPTH = 1

    CONFIGURE_FOR_PERFORMANCE = False

    SAVE = False

    def __init__(self,data_dir,test_size,configure_for_performance,height = 64,width = 64, depth =1 ):
        self.data_dir = pathlib.Path(data_dir)

        print(data_dir)
        if (data_dir.find('kaggle') != -1):
            self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.jpg')))
        elif (data_dir.find('WHOI') != -1):
            self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.png')))
        elif (data_dir.find('Luo') != -1):
            self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.jpg')))
        elif (data_dir.find('Pastore') != -1):
            self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.jpg')))
        elif (data_dir.find('Ailaron') != -1):
            self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.tiff')))

        else:
            print("Unknown data format. System exit.")
            sys.exit(0)


        print("found ", self.IMAGE_COUNT," images in folder")

        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "test"])
        print("found ",self.CLASS_NAMES," classes")

        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
        self.CONFIGURE_FOR_PERFORMANCE = configure_for_performance
        self.TEST_SIZE = test_size
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.IMG_DEPTH = depth


    def get_label(self,file_path):
        parts = tf.strings.split(file_path, '/')
        return parts[-2] == self.CLASS_NAMES

    def decode_img(self,img):

        print("test")
        img = tf.image.decode_jpeg(img, channels=self.IMG_DEPTH) #color images
        print("more test")
        #img = tf.image.convert_image_dtype(img, tf.float32)
        #convert unit8 tensor to floats in the [0,1]range
        #img = img/255
        #print(img)
        #import tensorflow_io as tfio

        #img = tfio.experimental.image.decode_tiff_info(
        #img, name=None
        #)

        return img #tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self,file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def configure_for_performance(self,ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        #ds = ds.batch(32)
        #ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def normalize(self,x, y):
        x = tf.image.per_image_standardization(x)
        return x, y



    def load_data(self):
        labeled_ds = self.list_ds.map(self.process_path) #, num_parallel_calls=AUTOTUNE)



        #If CONFIGURE_FOR_PERFORMANCE = True: return a tensorflow dataset with memory control
        if self.CONFIGURE_FOR_PERFORMANCE:
            return labeled_ds.shuffle(50000).map(self.normalize)#self.configure_for_performance(labeled_ds)


        yes = True
        #import tensorflow_datasets as tfds

        if yes:
            #ds = np.zeros((self.IMAGE_COUNT,self.IMG_HEIGHT,self.IMG_WIDTH,self.IMG_DEPTH))
            #ds_label = np.zeros(self.IMAGE_COUNT)

            #i = 0
            images = np.array([])
            labels = np.array([])
            if yes:
                #for image, label in (labeled_ds):
                #    images = np.append(images, np.array([image.numpy()])) #np.array([image.numpy()])
                #    labels = np.array([np.where(label.numpy())[0]])
                test = labeled_ds
                #print(test)
                images = np.array([(image.numpy(),label) for image, label in (test)])

                n = np.array([image for image, label in (images)])
                ds_label = np.zeros(self.IMAGE_COUNT)

                i = 0

                for image, label in (images):
                    ds_label[i]= np.where(label.numpy())[0]
                    i  = i + 1
                #labels = np.array([np.where(label.numpy())[0] for image, label in (images)])

                return n, ds_label

                #print(labels[0])
                #return images, labels
                #print("Error fafass importing images")
                #label = np.array([label for image, label in (labeled_ds)])
                #print("Error fafass importing images")
                #label = np.array([label for image, label in (labeled_ds)])


                #list = []
                #print(images.shape,images[0].shape[0],images[0].shape[1], images[0].shape[2])
                #for image, label in (labeled_ds):
                #    ds = np.zeros((images[i].shape[0],images[i].shape[1],images[i].shape[2]))

                    #list = np.append(list,image,axis = 0)

                    #ds_label[i]= np.where(label.numpy())[0]
                    #i  = i + 1

            #except:
            #    print("Error in importing images")
            #    sys.exit()

            #return list, ds_label

        #Else: Return a numpy array dataset
        ds = np.zeros((self.IMAGE_COUNT,self.IMG_HEIGHT,self.IMG_WIDTH,self.IMG_DEPTH))
        ds_label = np.zeros(self.IMAGE_COUNT)

        i = 0
        try:
            for image, label in (labeled_ds):

                #print(image.shape)
                ds[i] = image
                ds_label[i]= np.where(label.numpy())[0]
                i  = i + 1

        except:
            print("Error in importing images")
            sys.exit()

        if self.TEST_SIZE > 0:
            train, test, train_label, test_label = train_test_split(ds, ds_label, test_size=self.TEST_SIZE)
            print ("training data shape",train.shape, train_label.shape)
            print ("testing data shape",test.shape, test_label.shape)
            return train, train_label #, test, test_label
        else:
            train = ds
            train_label = ds_label
            test = 0
            test_label = 0
            print ("training data shape",train.shape, train_label.shape)
            print(train, train.mean())
            return train, train_label #, test, test_label

import glob
from PIL import Image
from pathlib import Path


### Similar class to load dataset, but for tiff images. To be replaced when tensorflow gets tiff support ###
class LoadDatasetTIFF():
    data_dir = None
    list_ds = None

    IMAGE_COUNT = None
    CLASS_NAMES = None
    TEST_SIZE = 0.1

    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_DEPTH = 1

    CONFIGURE_FOR_PERFORMANCE = False

    SAVE = False

    def __init__(self,data_dir,test_size,configure_for_performance,height = 64,width = 64, depth =1 ):
        self.data_dir = pathlib.Path(data_dir)

        if (data_dir.find('Ailaron') != -1):
            self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.tiff')))

        else:
            print("Unknown data format. System exit.")
            sys.exit(0)

        print("found ", self.IMAGE_COUNT," images in folder")

        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "test"])
        print("found ",self.CLASS_NAMES," classes")

        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
        self.CONFIGURE_FOR_PERFORMANCE = configure_for_performance
        self.TEST_SIZE = test_size
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.IMG_DEPTH = depth



    # Takes as input path to image file and returns
    # resized 3 channel RGB image of as numpy array of size (256, 256, 3)
    def getPic(self,img_path):
        return np.array(Image.open(img_path).convert('RGB'),dtype=np.float32) #.resize((224,224),Image.ANTIALIAS ))

    # returns the Label of the image based on its first 3 characters
    def get_label(self,img_path):
        #print(Path(img_path).absolute().parent.name)
        return Path(img_path).absolute().parent.name == self.CLASS_NAMES#Path(img_path).absolute().name[0:3]

    # Return the images and corresponding labels as numpy arrays
    def load_data(self,data_path):
        img_paths = list()
        # Recursively find all the image files from the path data_path
        for img_path in glob.glob(data_path+"/**/*"):
            #print(img_path)
            img_paths.append(img_path)

        images = [] #np.zeros((len(img_paths),256,256,3))
        print(len(images))
        labels = np.zeros((len(img_paths)))

        # Read and resize the images
        # Get the encoded labels
        #n = np.array([image for image, label in (images)])
        for i, img_path in enumerate(img_paths):
            #images[i] = self.getPic(img_path)
            if not images:
                images.append(self.getPic(img_path))
            else:
                images.append(self.getPic(img_path) )#np.append([images],[self.getPic(img_path)])

            labels[i] = np.where(self.get_label(img_path))[0]

        images = np.array([(image) for image in (images)])


        print("sad")
        #print(images)
        return images,labels

    def get_list_ds(self):
        return self.list_ds
