##################
#Tensorflow-keras based implementation of DeepCluster
#Using VGG16, 5Conv and ResNet as feature extractor

#This file is based on the Caron et al. DeepCluster from the following GitHub
#https://github.com/facebookresearch/deepcluster

#Note 1. that the code is changed from a pytorch implementation to a tensorflow implementation.
#Performance might therefore differ.

#Note 2. For an easy walkthrough see https://amitness.com/2020/04/deepcluster/

"""
Input: 1. arch: which model architecture to use: 5Conv, VGG16 or ResNet152
"""
##################



import tensorflow as tf
from skimage.transform import resize
import cv2
import numpy as np

# Arguments to perform data preprocessing and data augmentation in preprocessing pipeline
DATA_GEN_ARGS = dict(#featurewise_center=True,
                 #featurewise_std_normalization=True,
                 rotation_range=90,
                 width_shift_range=0.1, #0,1
                 height_shift_range=0.1, #0,1
                 zoom_range=0.2, #0,2
                 #UNDER WAS ADDED LAST
                 horizontal_flip = True,
                 vertical_flip = True,
                 #zca_whitening=True,
                 )

# Arguments to perform data preprocessing in preprocessing pipeline
DATA_GEN_ARGS_VAL = dict(#featurewise_center=True,
                 #featurewise_std_normalization=True,
                 #UNDER WAS ADDED LAST
                 #zca_whitening=True
                 )



class Preprocessing():
    DATASET='Kaggle'
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64
    IMAGE_DEPTH = 3
    NUM_CLASSES = 121
    AUTOENCODER = False
    BATCH_SIZE = 32

    data = None
    label = None


    def __init__(self, data, label,dataset='Kaggle', num_classes = 121,input_shape = (64,64,3), autoencoder = False, batch_size = 32):
            self.DATASET = dataset
            self.NUM_CLASSES = num_classes
            self.IMAGE_WIDTH = input_shape[0]
            self.IMAGE_HEIGHT = input_shape[1]
            self.AUTOENCODER = autoencoder
            self.BATCH_SIZE = batch_size

            if dataset == 'Ailaron':
                self.IMAGE_WIDTH = 64
                self.IMAGE_HEIGHT = 64
                self.IMAGE_DEPTH = 3
                self.NUM_CLASSES = 7

            self.createPreprocessedDataset(data, label)

            return


    def createPreprocessedDataset(self,data, label):

        data = np.array([cv2.resize(img, dsize=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR) for img in (data)])


        if self.DATASET == 'Kaggle':
            print(data.mean())
            #Make data type float 32
            data = data.astype(np.float32)
            #Center data
            #data = data - data.mean()
            #Make data between [-1-1] ##NOTICED THAT THIS SPEEDS UP TRAINING##
            #data = data/255
            data = (data - 127.5) / 127.5

            #Make depth = 3
            data=np.stack([data]*3, axis=-1)
            data = data.reshape(-1, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH)

        elif self.DATASET == 'Ailaron':
            #Make data type float 32
            data = data.astype(np.float32)
            #Center data
            data = data - 195.18
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            data = data/255

        if self.AUTOENCODER:
            self.data = data
            self.label = data

        else:
            self.data = data
            self.label = label
        from utils import visualize_input
        #visualize_input(data[0:16])

        print('Data min=%.3f, max=%.3f' % (data.min(), data.max()))

    def returnAugmentedDataset(self):
        ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS)


        gen = ImageDataGenerator.flow(
            self.data,
            y=self.label,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            #sample_weight=None,
            #seed=None,
            #save_to_dir=None,
            #save_prefix="",
            #save_format="png",
            #subset=None,
            )

        if tf.__version__== '1.15.0':
            print('yo')
            return gen

        print('no')
        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: gen,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH],[None,self.NUM_CLASSES])
        )
        return ds

    def returnImages(self):
        return self.data

    def returnTrainDataset(self):
        return tf.data.Dataset.from_tensor_slices((self.data, self.label)).batch(self.BATCH_SIZE).shuffle(60000)

    def returnDataset(self):
        if tf.__version__== '1.15.0':
            
            return self.data,self.label

        return tf.data.Dataset.from_tensor_slices((self.data, self.label)).batch(self.BATCH_SIZE)

    def updateLabels(self,label):
        self.label = label

    def shape(self):
        return self.data.shape

class PreprocessingFromDataframe(Preprocessing):
    predictgen = None
    traingen = None


    def __init__(self,data,dataset='Kaggle', num_classes = 121, input_shape=(64,64,3)):
            self.DATASET = dataset
            self.NUM_CLASSES = num_classes
            self.IMAGE_WIDTH = input_shape[0]
            self.IMAGE_HEIGHT = input_shape[1]


            if dataset == 'Ailaron':
                self.IMAGE_WIDTH = 64
                self.IMAGE_HEIGHT = 64
                self.IMAGE_DEPTH = 3
                self.NUM_CLASSES = 7
            self.createPreprocessedGenerators(data)
            return

    def createPreprocessedGenerators(self,data):
        self.predictgen= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
        featurewise_center	= True,
        )

        self.traingen= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
        featurewise_center	= True,
        rotation_range=90,
        width_shift_range=0.1, #0,1
        height_shift_range=0.1, #0,1
        zoom_range=0.2, #0,2
        #UNDER WAS ADDED LAST
        horizontal_flip = True,
        vertical_flip = True)

        data = np.array([cv2.resize(img, dsize=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR) for img in (data)])
        #data=np.stack([data]*3, axis=-1)
        data = data.reshape(-1, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,1)

        self.predictgen.fit(data)
        self.traingen.fit(data)

    def createPreprocessedDataset(self,filename= ""):
        import pandas as pd
        df=pd.read_csv(filename)
        self.NUM_CLASSES = len(df.label.unique())

        predict_generator= self.predictgen.flow_from_dataframe(dataframe=df, directory="", x_col="id", y_col="label", class_mode="categorical",
        target_size=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT), batch_size=32, shuffle=False, color_mode='rgb', interpolation = "bilinear")


        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: predict_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3],[None,self.NUM_CLASSES])
        )

        return ds

    def createPreprocessedAugmentedDataset(self, filename = ""):
        import pandas as pd
        df=pd.read_csv(filename)
        self.NUM_CLASSES = len(df.label.unique())

        train_generator=self.traingen.flow_from_dataframe(dataframe=df, directory="", x_col="id", y_col="label", class_mode="categorical",
        target_size=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT), batch_size=32, shuffle =True, color_mode='rgb', interpolation = "bilinear",save_to_dir = 'test',save_format = 'png')

        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3],[None,self.NUM_CLASSES])
        )
        return ds

    def returnGenerator(self):
        return self.predictgen, self.traingen

    def returnImages(self, filename = ""):
        import pandas as pd
        from PIL import Image

        df=pd.read_csv(filename)
        #df.columns = ['id',"label"]
        #print(df.head())
        images = []
        ds = np.zeros((3720,64,64))
        for index, row in df.iterrows():
            img = np.array(Image.open(row['id']),dtype=np.float32)
            images.append(img)

        data = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (images)])

        return data

    def returnLabels(self, filename = ""):
        import pandas as pd
        df=pd.read_csv(filename)

        labels = np.zeros(len(df.label),dtype='int')

        for i, val in enumerate(df.label):
            labels[i] = (int(val[1]))

        return labels

    def updateImageSize(self, image_width, image_height):
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        return

    def updateNumClasses(self,num):
        self.NUM_CLASSES = num
        return

    def returnDatasetSize(self, filename = ""):
        import pandas as pd
        df=pd.read_csv(filename)
        #df.columns = ['id',"label"]
        print(df.head())

        labels = []
        df2 = df.iloc[:, 1]
        list = df2.to_numpy()
        for val in list:
            labels.append(int(val[1]))
        labels = np.array(labels)

        return labels.shape[0]

    def shape(self):
        return




def get_label(file_path, classes):
    parts = file_path.split('/')
    return parts[-2] == classes

def makeCSVFile(path, filename = '',output_filename = '',   split = 0):
    import os
    import csv
    import pathlib
    import glob

    # Returns a list of names in list files.
    files = glob.glob(filename+'//**/*.jpg',
                       recursive = True)

    filepath = pathlib.Path(filename+"/")

    data = []
    labels = []
    classes = np.array([item.name for item in filepath.glob('*') if item.name != "test"])
    print('Found ',classes, ' classes of images in folder')

    for file in files:
        label = np.where(get_label(file,classes))[0]
        data.append([file,label])
        labels.append(label)

    if split == 0:
        with open(output_filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["id","label"])
            for d in data:
                writer.writerow(d)
    elif split > 0 and split < 1:
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(data,  test_size=split, stratify = labels)

        with open('trainLabels.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["id","label"])
            for d in train:
                writer.writerow(d)

        with open('testLabels.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["id","label"])
            for d in test:
                writer.writerow(d)














































"""
    def createPreprocessedDatasetasa(self, val_data,val_label):
        ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS_VAL)
        #ImageDataGenerator.fit(data)
        val_data = np.array([cv2.resize(img, dsize=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR) for img in (val_data)])

        if self.DATASET == 'Kaggle':
            #Make data type float 32
            val_data = val_data.astype(np.float32)
            #Center data
            val_data = val_data - 239.64
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            val_data = val_data/255
            #Make depth = 3
            val_data=np.stack([val_data]*3, axis=-1)
            val_data = val_data.reshape(-1, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH)

        elif self.DATASET == 'Ailaron':
            #Make data type float 32
            val_data = val_data.astype(np.float32)
            #Center data
            val_data = val_data - 195.18
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            val_data = val_data/255

        #for i in range(len(val_data)):
        #    val_data[i] = ((val_data[i] - 239)/255) #.astype(np.float32) FOR KAGGLE 239.64
        #Subtract mean
        #val_data = val_data - 195.18 #FOR AILARON 195.18, FOR KAGGLE 239.64

        #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
        #val_data = val_data/255

        #Convert to float32
        #val_data = val_data.astype(np.float32)

        #Make depth = 3
        #val_data=np.stack([val_data]*3, axis=-1)
        #val_data = val_data.reshape(-1, 64,64,3)

        gen = ImageDataGenerator.flow(
            val_data,
            y=val_label,
            batch_size=32,
            shuffle=True,
            #sample_weight=None,
            #seed=None,
            #save_to_dir=None,
            #save_prefix="",
            #save_format="png",
            #subset=None,
            )


        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: gen,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH],[None,self.NUM_CLASSES])
        )

        return ds

    def createPreprocessedAugmentedDataset(self,train_data,train_label):
        ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS)
        #ImageDataGenerator.fit(data)


        from utils import visualize_input

        train_data = np.array([cv2.resize(img, dsize=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR) for img in (train_data)])

        print(train_data.mean())
        #train_data = train_data.astype(np.float32)
        print(train_data.shape)
        print('train_data min=%.3f, max=%.3f' % (train_data.min(), train_data.max()))
        #train_data = (train_data - 195.18) #FOR AILARON 195.18, FOR KAGGLE 239.64
        #for img in (train_data):
        #    print(img)
        #    train_data[i] = ((train_data[i] - 239)/255).astype(np.float32)

        if self.DATASET == 'Kaggle':
            #Make data type float 32
            train_data = train_data.astype(np.float32)
            #Center data
            train_data = train_data - 239.64
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            train_data = train_data/255
            #Make depth = 3
            train_data=np.stack([train_data]*3, axis=-1)
            train_data = train_data.reshape(-1, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH)

        elif self.DATASET == 'Ailaron':
            #Make data type float 32
            train_data = train_data.astype(np.float32)
            #Center data
            train_data = train_data - 195.18
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            train_data = train_data/255

        visualize_input(train_data[0:16])

        #train_data = train_data/255

        #train_data = train_data.astype(np.float32)

        print('train_data min=%.3f, max=%.3f' % (train_data.min(), train_data.max()))

        #train_data=np.stack([train_data]*3, axis=-1)
        #train_data = train_data.reshape(-1, 64,64,3)

        print(train_data.dtype)
        gen = ImageDataGenerator.flow(
            train_data,
            y=train_label,
            batch_size=32,
            shuffle=True,
            #sample_weight=None,
            #seed=None,
            #save_to_dir=None,
            #save_prefix="",
            #save_format="png",
            #subset=None,
            )

        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: gen,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH],[None,self.NUM_CLASSES])
        )
        return ds

    def createPreprocessedPredictDataset(self,train_data,train_label):


        train_data = np.array([cv2.resize(img, dsize=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR) for img in (train_data)])

        print(train_data.mean())
        #train_data = (train_data - 239.64)

        #train_data = train_data/255
        #ds = np.zeros((30336,224,224,3))
        #for i, img in enumerate(train_data):
        #    ds[i] = ((img.astype(np.float32) - 239)/255)

        #train_data = ds
        #ds = 0

        if self.DATASET == 'Kaggle':
            #Make data type float 32
            train_data = train_data.astype(np.float32)
            #Center data
            train_data = train_data - 239.64
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            train_data = train_data/255
            #Make depth = 3
            train_data=np.stack([train_data]*3, axis=-1)
            train_data = train_data.reshape(-1, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH)

        elif self.DATASET == 'Ailaron':
            #Make data type float 32
            train_data = train_data.astype(np.float32)
            #Center data
            train_data = train_data - 195.18
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            train_data = train_data/255

        from utils import visualize_input
        visualize_input(train_data[0:16])
        #train_data = train_data.astype(np.float32)

        print('train_data min=%.3f, max=%.3f' % (train_data.min(), train_data.max()))

        #train_data=np.stack([train_data]*3, axis=-1)
        #train_data = train_data.reshape(-1, 224,224,3)

        return tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(32)



        return ds
"""
