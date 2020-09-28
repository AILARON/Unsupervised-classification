



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
    NUM_CLASSES = 200


    data = None
    label = None


    def __init__(self, data, label,dataset='Kaggle', num_classes = 121):
            self.DATASET = dataset
            self.NUM_CLASSES = num_classes

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
            data = data - 239.64
            #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
            data = data/255
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

        self.data = data
        self.label = label

        from utils import visualize_input
        visualize_input(data[0:16])

        print('Data min=%.3f, max=%.3f' % (data.min(), data.max()))

    def returnAugmentedDataset(self):
        ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS)
        

        gen = ImageDataGenerator.flow(
            self.data,
            y=self.label,
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

    def returnDataset(self):
        return tf.data.Dataset.from_tensor_slices((self.data, self.label)).batch(32)

    def updateLabels(self,label):
        self.label = label

class PreprocessingFromDataframe(Preprocessing):

    def __init__(self,dataset='Kaggle', num_classes = 121):
            self.DATASET = dataset
            self.NUM_CLASSES = num_classes

            if dataset == 'Ailaron':
                self.IMAGE_WIDTH = 64
                self.IMAGE_HEIGHT = 64
                self.IMAGE_DEPTH = 3
                self.NUM_CLASSES = 7

            return
    def createPreprocessedDataset(self):
        import pandas as pd
        df=pd.read_csv(r"output.csv")
        #df.columns = ['id',"label"]
        print(df.head())

        datagen= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_generator=datagen.flow_from_dataframe(dataframe=df, directory="", x_col="id", y_col="label", class_mode="categorical",
        target_size=(224,224), batch_size=32, shuffle=False)

        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH],[None,self.NUM_CLASSES])
        )
        return ds

    def createPreprocessedAugmentedDataset(self):
        import pandas as pd
        df=pd.read_csv(r"output.csv")
        #df.columns = ['id',"label"]
        print(df.head())

        datagen= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,rotation_range=90,
        width_shift_range=0.1, #0,1
        height_shift_range=0.1, #0,1
        zoom_range=0.2, #0,2
        #UNDER WAS ADDED LAST
        horizontal_flip = True,
        vertical_flip = True)

        train_generator=datagen.flow_from_dataframe(dataframe=df, directory="", x_col="id", y_col="label", class_mode="categorical",
        target_size=(224,224), batch_size=32, shuffle =True)

        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.IMAGE_DEPTH],[None,self.NUM_CLASSES])
        )
        return ds

    def createPreprocessedTestDataset(self):
        import pandas as pd
        df=pd.read_csv(r"test.csv")
        #df.columns = ['id',"label"]
        print(df.head())

        labels = []
        df2 = df.iloc[:, 1]
        list = df2.to_numpy()
        for val in list:
            labels.append(int(val[1]))
        labels = np.array(labels)



        datagen= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_generator=datagen.flow_from_dataframe(dataframe=df, directory="", x_col="id", y_col="label",
         class_mode="categorical",color_mode='grayscale',target_size=(224,224), batch_size=32, shuffle=False)

        # Wrap the generator with tf.data
        ds = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,1],[None,5])
        )



        return ds, labels


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
