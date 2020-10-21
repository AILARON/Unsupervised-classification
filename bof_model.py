#SPP imporvement loss: 0.6062 - accuracy: 0.7873 - val_loss: 1.0111 - val_accuracy: 0.7121



from __future__ import print_function
#from dataset import load_mnist, resize_images
import tensorflow as tf


from cbof import BoF_Pooling, initialize_bof_layers
from spp import SpatialPyramidPooling

from preprocessing import Preprocessing, PreprocessingFromDataframe


import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist():
    """
    Loads the MNIST dataset
    :return:
    """
    img_rows, img_cols = 28, 28
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def load_kaggle():

    img_rows, img_cols = 64, 64
    num_classes = 121

    from load_dataset import importKaggleTrain
    data, labels = importKaggleTrain()
    #data = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (data)])
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    #x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    #x_train = x_train.astype('float32') #/ 255
    #x_test = x_test.astype('float32') #/ 255

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def resize_images(x, scale=0.8):
    """
    Resizes a collection of grayscale images
    :param x: list of images
    :param scale: the scale
    :return:
    """
    img_size = [int(x * scale) for x in x.shape[1:]]
    new_data = np.zeros((x.shape[0], img_size[0], img_size[1], x.shape[3]))
    for k in range(x.shape[0]):
        new_data[k, :, :, 0] = cv2.resize(x[k][:, :, 0], (img_size[0], img_size[1]))
    return np.float32(new_data)

def build_model(pool_type='max', n_output_filters=32, n_codewords=32):
    if pool_type == 'max':
        input_size = None
    else:
        input_size = None

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(2, 2),padding="same", input_shape=(input_size, input_size, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=(2, 2),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same"))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=(2, 2),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(512, kernel_size=(2, 2),padding="same"))



    if pool_type == 'max':
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
    elif pool_type == 'gmp':
        model.add(tf.keras.layers.GlobalMaxPool2D())
    elif pool_type == 'spp':
        model.add(SpatialPyramidPooling([1, 2,3]))
    elif pool_type == 'bof':
        model.add(BoF_Pooling(n_codewords, spatial_level=0))
    elif pool_type == 'spatial_bof':
        model.add(BoF_Pooling(n_codewords, spatial_level=1))
    else:
        assert Flatten

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(121, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])




    return model


def COAPNET(input_shape = (None,None,3), output_shape = 121, include_top = True):
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
    #x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    #x = tf.keras.layers.Activation('relu')(x)
    #x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    if include_top == False:
        return tf.keras.models.Model(inputs, x,name="SPP")
    else:
        x = SpatialPyramidPooling([1, 2,3])(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

        return tf.keras.models.Model(inputs, x,name="SPP")


def evaluate_model(pool_type, n_filters=128, n_codewords=0):
    x_train, y_train, x_test, y_test = load_kaggle()

    if pool_type == 'bof' or pool_type == 'spatial_bof':

        preprocess_training = Preprocessing(x_train,y_train)
        preprocess_test = Preprocessing(x_test, y_test)

        dataset_train = preprocess_training.returnAugmentedDataset()
        dataset_test = preprocess_test.returnDataset()

        print("Evaluating model: ", pool_type)
        model = build_model(pool_type, n_output_filters=n_filters, n_codewords=n_codewords)
        model.fit(dataset, epochs=100, verbose=1,shuffle=True, steps_per_epoch=27302//32)

    elif pool_type == 'spp':
        training_type = ['variable_size','variable_size_batch','variable_size_epoch']
        type = training_type[2]

        if type == 'variable_size':
            #Train network using spp model on variable size images

            #preprocess dataset
            for i, image in enumerate(x_train):
                x_train[i] = (image)/255
            # make tf dataset
            import itertools
            dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(x_train, y_train),
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([None, None, 1]),
                                                             tf.TensorShape([None])))

            # train model
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.09)

            epochs = 2
            for epoch in range(epochs):
                print("\nStart of epoch %d" % (epoch,))

                # Iterate over the batches of the dataset.
                for step, (x_batch_train, y_batch_train) in enumerate(dataset):


                    # Open a GradientTape to record the operations run
                    # during the forward pass, which enables auto-differentiation.
                    with tf.GradientTape() as tape:

                        # Run the forward pass of the layer.
                        # The operations that the layer applies
                        # to its inputs are going to be recorded
                        # on the GradientTape.
                        logits = model(x_batch_train, training=True)  # Logits for this minibatch

                        # Compute the loss value for this minibatch.
                        loss_value = loss_fn(y_batch_train, logits)

                    # Use the gradient tape to automatically retrieve
                    # the gradients of the trainable variables with respect to the loss.
                    grads = tape.gradient(loss_value, model.trainable_weights)

                    # Run one step of gradient descent by updating
                    # the value of the variables to minimize the loss.
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

                    # Log every 200 batches.
                    if step % 10 == 0:
                        print(
                            "Training loss (for one batch) at step %d: %.4f"
                            % (step, float(loss_value))
                        )
                        print("Seen so far: %s samples" % ((step + 1) ))

        if type == 'variable_size_batch':

            #Train network using spp model on variable size images batches

            #preprocess dataset
            for i, image in enumerate(x_train):
                x_train[i] = (image)/255

            for i, image in enumerate(x_test):
                x_test[i] = (image)/255
            # make tf dataset
            import itertools
            dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(x_train, y_train),
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([None, None, 1]),
                                                             tf.TensorShape([None])))

            dataset_test = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(x_test, y_test),
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([None, None, 1]),
                                                             tf.TensorShape([None])))

            dataset=dataset.padded_batch(32,padded_shapes=([None,None,1],[None])).shuffle(40000).repeat(count=100)

            dataset_test=dataset_test.padded_batch(32,padded_shapes=([None,None,1],[None])).shuffle(10000).repeat(count=100)


            iterator = dataset.make_one_shot_iterator()

            print((iterator.get_next()))  # ==> '[4, 2]'
            print((iterator.get_next()))  # ==> '[3, 4, 5]'

            print("Evaluating model: ", pool_type)
            model = build_model(pool_type, n_output_filters=n_filters, n_codewords=n_codewords)
            model.fit(dataset, epochs=100, verbose=1,shuffle=True, steps_per_epoch=27302//32, validation_data =dataset_test,validation_steps= x_test.shape[0] // 32)


        if type == 'variable_size_epoch':
            preprocess = PreprocessingFromDataframe(x_train,y_train,image_width = 64, image_height = 64)
            epochs = 100
            import random
            #model = build_model(pool_type, n_output_filters=n_filters, n_codewords=n_codewords)
            model = COAPNET()
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
            stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.01, patience=5, verbose=1, mode='auto',
            baseline=None, restore_best_weights=False)

            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
            for i in range(epochs):
                print('epoch',i)
                image_dim = random.randint(100, 160)
                print(image_dim)
                if image_dim % 2 == 1:
                    image_dim += 1
                preprocess.updateImageSize(image_dim,image_dim)

                dataset_train = preprocess.createPreprocessedAugmentedDataset()
                dataset_test = preprocess.createPreprocessedDataset()

                print("Evaluating model: ", pool_type)

                model.fit(dataset_train, epochs=1, verbose=1,shuffle=True, steps_per_epoch=27302//32,validation_data =dataset_test,
                validation_steps= x_test.shape[0] // 32,callbacks =[stopping])

            saveWeights(model)

def saveWeights(model):
    # Save JSON config to disk
    json_config = model.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(json_config)
        # Save weights to disk
        print("[Info] saving weights")
        model.save_weights(str(model.name)+"_"+"baseline"+'_weights.h5')

def loadWeights(model):
    print("[Info] loading previous weights")
    try:
        model.load_weights(str(model.name)+"_"+"baseline"+'_weights.h5')
    except:
        print("Could not load weights")

    return model


def example():


    # Baseline model
    #evaluate_model('max', n_filters=64)
    #evaluate_model('max', n_filters=32)
    #evaluate_model('max', n_filters=4)

    # Global Max Pooling
    #evaluate_model('gmp', n_filters=16)
    #evaluate_model('gmp', n_filters=24)
    #evaluate_model('gmp', n_filters=64)
    #evaluate_model('gmp', n_filters=128)

    # SPP
    #evaluate_model('spp', n_filters=8)
    #evaluate_model('spp', n_filters=16)
    #evaluate_model('spp', n_filters=32)
    evaluate_model('spp', n_filters=64)

    # CBoF
    #evaluate_model('bof', n_filters=32, n_codewords=16)
    #evaluate_model('bof', n_filters=32, n_codewords=64)
    #evaluate_model('bof', n_filters=32, n_codewords=128)
    #evaluate_model('bof', n_filters=32, n_codewords=256)

    # Spatial CBoF
    #evaluate_model('spatial_bof', n_filters=32, n_codewords=8)
    #evaluate_model('spatial_bof', n_filters=32, n_codewords=16)
    #evaluate_model('spatial_bof', n_filters=64, n_codewords=32)
    #evaluate_model('spatial_bof', n_filters=32, n_codewords=64)
    #evaluate_model('spatial_bof', n_filters=64, n_codewords=128)
    #evaluate_model('spatial_bof', n_filters=32, n_codewords=256)
