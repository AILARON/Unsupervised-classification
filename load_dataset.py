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


KAGGLE_ORIGINAL_TRAIN = "dataset/kaggle_original/"
KAGGLE_ORIGINAL_TEST = "dataset/kaggle_five_classes/"
WHOI_ORIGINAL_TRAIN_2014 = "dataset/WHOI/2014/"
WHOI_ORIGINAL_TRAIN_2013 = "dataset/WHOI/2013/"
LUOTRAINING = "dataset/Luo_Training/"
PASTORETRAINING ="dataset/Pastore_Training"
AILARONTRAINING = "dataset/Ailaron_Training"
AILARONTEST = "dataset/Ailaron_Test"

DATA_FORMAT = ["jpg","png"]

def importWHOI():
    loader = LoadDataset(WHOI_ORIGINAL_TRAIN_2014)
    return loader.load_data()

def importKaggle(train = True):
        if train:
            loader = LoadDataset(KAGGLE_ORIGINAL_TRAIN)
        else:
            loader = LoadDataset(KAGGLE_ORIGINAL_TEST)
        return loader.load_data()

def importLuoTraining():
        loader = LoadDataset(LUOTRAINING)
        return loader.load_data()

def importPastoreTraining():
        loader = LoadDataset(PASTORETRAINING)
        return loader.load_data()

def importAilaron(train = True):
    if train:
        loader = LoadDatasetTIFF(AILARONTRAINING)
    else:
        loader = LoadDatasetTIFF(AILARONTEST)
    return loader.load_data()

def importKaggleOld(train=True):
    if train:
        loader = LoadDatasetOld(KAGGLE_ORIGINAL_TRAIN)
        return loader.load_data(KAGGLE_ORIGINAL_TRAIN)
    else:
        loader = LoadDatasetOld(KAGGLE_ORIGINAL_TEST)
        return loader.load_data(KAGGLE_ORIGINAL_TEST)
    return
class LoadDataset:
    data_dir = None
    list_ds = None

    IMAGE_COUNT = None
    CLASS_NAMES = None
    TEST_SIZE = 0

    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_DEPTH = 1

    CONFIGURE_FOR_PERFORMANCE = False
    SAVE = False

    def __init__(self,data_dir,height = 64,width = 64, depth =1 ):
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

        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.IMG_DEPTH = depth


    def get_label(self,file_path):
        parts = tf.strings.split(file_path, '/')
        return parts[-2] == self.CLASS_NAMES

    def decode_img(self,img):
        img = tf.image.decode_jpeg(img, channels=self.IMG_DEPTH)
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

        images = np.array([])
        labels = np.array([])

        ds = labeled_ds

        images = np.array([(image.numpy(),label) for image, label in (ds)])

        data = np.array([image for image, label in (images)])
        data_label = np.zeros(self.IMAGE_COUNT)

        i = 0

        for image, label in (images):
            data_label[i]= np.where(label.numpy())[0]
            i  = i + 1

        return data, data_label



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

    def __init__(self,data_dir,height = 64,width = 64, depth =1 ):
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

        return images,labels

    def get_list_ds(self):
        return self.list_ds



### Similar class to load dataset, but for tf 1.x ###
class LoadDatasetOld():
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

    def __init__(self,data_dir,height = 64,width = 64, depth =1 ):
        self.data_dir = pathlib.Path(data_dir)

        if (data_dir.find('kaggle') != -1):
            self.IMAGE_COUNT = len(list(self.data_dir.glob('*/*.jpg')))

        else:
            print("Unknown data format. System exit.")
            sys.exit(0)

        print("found ", self.IMAGE_COUNT," images in folder")

        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "test"])
        print("found ",self.CLASS_NAMES," classes")

        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))


        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.IMG_DEPTH = depth



    # Takes as input path to image file and returns
    # resized 3 channel RGB image of as numpy array of size (256, 256, 3)
    def getPic(self,img_path):
        return np.array(Image.open(img_path),dtype=np.float32) #.resize((224,224),Image.ANTIALIAS ))

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



def loadFromDataFrame(filename= ""):
    import pandas as pd
    from PIL import Image

    df=pd.read_csv(filename)

    print(len(df.index))

    images = []
    labels = np.zeros(len(df.label),dtype='int')
    for index, row in df.iterrows():
        img = np.array(Image.open(row['id']),dtype=np.float32)
        images.append(img)

    for i, val in enumerate(df.label):
        labels[i] = (int(val[1]))

    return images, labels



def get_label(file_path, classes):
    parts = file_path.split('/')
    return parts[-2] == classes

def createCSVFile(filename,output_filename,drop_class = False):
    import os
    import csv
    import pathlib
    import glob
    import csv

    print(pathlib.Path("dataset/"+filename+"/").glob('*/*.jpg'))

    # Returns a list of names in list files.
    print("Using glob.glob()")
    files = glob.glob("dataset/"+filename+"//**/*.jpg",
                       recursive = True)

    filepath = pathlib.Path("dataset/"+filename+"/")

    data = []
    classes = np.array([item.name for item in filepath.glob('*') if item.name != "test"])
    print(classes)
    for file in files:
        print(file)
        if drop_class == True:
            if (file.find('copepod') == -1):
                label = np.where(get_label(file,classes))[0]
                print(label)
                data.append([file,label])
        else:
            label = np.where(get_label(file,classes))[0]
            print(label)
            data.append([file,label])


    with open(output_filename+'.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id","label"])
        for d in data:
            writer.writerow(d)

    return


def checkCSVFile(filename):
    import pandas as pd
    df=pd.read_csv(filename)
    df.columns = ['id',"label"]
    print(df.head())

    datagen= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator=datagen.flow_from_dataframe(dataframe=df, directory="", x_col="id", y_col="label", class_mode="categorical", target_size=(32,32), batch_size=32)
