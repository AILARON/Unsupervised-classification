#SPP imporvement loss: 0.6062 - accuracy: 0.7873 - val_loss: 1.0111 - val_accuracy: 0.7121



from __future__ import print_function
#from dataset import load_mnist, resize_images
import tensorflow as tf


from cbof import BoF_Pooling, initialize_bof_layers
from spp import SpatialPyramidPooling

from preprocessing import Preprocessing


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

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(2, 2),padding="same", input_shape=(input_size, input_size, 1)))
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
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.001),
                  metrics=['accuracy'])




    return model


def evaluate_model(pool_type, n_filters=128, n_codewords=0):
    x_train, y_train, x_test, y_test = load_kaggle()

    #preprocess_training = Preprocessing(x_train,y_train)
    #preprocess_test = Preprocessing(x_test, y_test)

    #dataset_train = preprocess_training.returnAugmentedDataset()
    #dataset_test = preprocess_test.returnDataset()

    #preprocess data:

    #x_train = np.array([np.expand_dims((image-239)/255,0) for image in (x_train)])


    test = []
    for i, image in enumerate(x_train):
        x_train[i] = (image)/255
        #test.append(np.expand_dims(x_train[i], axis=0))

    #print(test)
    #x_train = np.array(test)

    #print(x_train.shape)
    #print(x_train[0].shape)

    import itertools
    dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(x_train, y_train),
                                      output_types=(tf.float32, tf.float32),
                                      output_shapes=(tf.TensorShape([None, None, 1]),
                                                     tf.TensorShape([None])))
    dataset=dataset.padded_batch(32,padded_shapes=([None,None,1],[None]))
    #dataset = tf.data.Dataset.from_generator(lambda: x_train, tf.float32, output_shapes=[None,None,1])

    iterator = dataset.make_one_shot_iterator()


    #rt=tf.ragged.constant(x_train)
    #dataset = tf.data.Dataset.from_tensor_slices(rt).batch(32)


    print((iterator.get_next()))  # ==> '[4, 2]'
    print((iterator.get_next()))  # ==> '[3, 4, 5]'

    #print(list(dataset.take(3)))

    #for i, element in enumerate(dataset):
    #    print(element.shape)

    print("Evaluating model: ", pool_type)
    model = build_model(pool_type, n_output_filters=n_filters, n_codewords=n_codewords)
    model.fit(dataset, epochs=1, verbose=1,shuffle=True, steps_per_epoch=27302//32)
    #initialize_bof_layers(model, preprocess_training.returnImages())

    #for i in range(len(model.layers)-1):
    #    model.layers[i].trainable = False

    #model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.001),
    #              metrics=['accuracy'])

    #print(model.summary())
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

    for i, element in enumerate(dataset):
        model.fit(dataset, epochs=1, verbose=1,shuffle=True, steps_per_epoch=27302//32)

    #for i in range(len(model.layers)-1):
    #    model.layers[i].trainable = True

    #model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.01),
    #              metrics=['accuracy'])


    #print(model.summary())
    #model.fit(dataset_train, epochs=300, verbose=1,shuffle=True, validation_data=dataset_test, steps_per_epoch= x_train.shape[0] // 32,
    #validation_steps= x_test.shape[0] // 32)

    acc = 100 * model.evaluate(x_test, y_test, verbose=0)[1]
    print('Test error:', 100 - acc)

    if pool_type != 'max':
        acc1 = 100 * model.evaluate(resize_images(x_test, scale=0.8), y_test, verbose=0)[1]
        print("Test error (scale=0.8): ", 100 - acc1)


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
