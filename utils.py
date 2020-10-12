#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the utils
# Author: Aya Saad
# Date created: 24 September 2019
# Changed for this implementation on 3 March 2020 by Eivind Salvesen
#################################################################################################################

import os
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects as PathEffects
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
import seaborn as sns
from matplotlib import patheffects as PathEffects
import cv2
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K

color_list = ['r', 'b', 'g', 'c', 'k', 'y','m','b','w']
box_color_list = ['red', 'blue', 'green', 'cyan', 'yellow','magenta','black','white']

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs(config):
    for path in [config.output_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

# Utility function to visualize the outputs of PCA and t-SNE
def fashion_scatter(x, colors,save_name,save_data_dir):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))

    #palette = 10*color_list
    # palette = np.array(sns.color_palette(color_palette, num_classes))
    palette = np.array(sns.color_palette("hls", num_classes))
    sns.set_palette(palette)

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.set_title("TSNE", fontsize=20)
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=20)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    #Make save folder:
    os.makedirs(os.path.join(save_data_dir), exist_ok=True)

    #Save figure
    plt.savefig(os.path.join(save_data_dir, save_name+'_cluster.png'))
    plt.close()
    return

def tile_scatter(x, input_data,colors,save_name, save_data_dir):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    sns.set_palette(palette)

    # make figure
    f = plt.figure(figsize=(16, 12))
    plt.title("TSNE",fontsize = 20)
    #####
    tx, ty = x[:, 0], x[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    plt.plot(tx, ty, '.')
    width = 1000
    height = 1000
    max_dim = 100
    full_image = Image.new('RGB', (width, height), (255, 255, 255))

    for img, x, y, c in tqdm(zip(input_data, tx, ty, colors)):
        #tile = Image.open(img).convert('RGB')
        tile = Image.fromarray(np.uint8(img)).convert('RGB') #Image.fromarray(np.uint8(img * 255)).convert('RGB')
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((max(1, int(tile.width / rs)), max(1, int(tile.height / rs))), Image.ANTIALIAS)

        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tile.size[0] - 1, tile.size[1] - 1], fill=None, outline=box_color_list[int(c)], width=5)

        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)))

    #####

    #Make save folder:
    os.makedirs(os.path.join(save_data_dir), exist_ok=True)

    # Save full image
    plt.imshow(full_image)
    plt.savefig(os.path.join(save_data_dir, save_name+'_image.png'))
    plt.close()
    return


def visualize_input(image_data, name ="name"):
    #cv2.imwrite(name+".png",image_data)
    #return
    output = "4x4"
    if output == "4x4":
        hstack = None
        vstack = None
        for i in range (4):
            hstack = None
            for j in range (4):
                print(i*4 +j)
                original = (image_data[i*4 +j] +1)*127 #+239

                if hstack is None:
                    hstack = original
                else:
                    hstack = np.hstack([hstack, original])
            if vstack is None:
                vstack = hstack
            else:
                vstack = np.vstack([vstack, hstack])
        cv2.imwrite(name+".png",vstack)
        #im = Image.fromarray(vstack)
        #im.save('name.jpg', format='JPEG', quality=100)
        return

    outputs = None
    for i in range(0, min(10, len(image_data))):
        # retrieve one original image and its recovered counterpart
    	original = image_data[i] #*255) .astype("uint8")

    	# if the outputs array is empty, initialize it as the current
    	if outputs is None:
    		outputs = original

    	# otherwise, vertically stack the outputs
    	else:
    		outputs = np.vstack([outputs, original])

    #save to file
    cv2.imwrite(name+".png",outputs)


def visualize_input_output(model,image_data,name ="name"):

    if name == "noisy":
        noise_factor = 0.05
        image_data = cv2.normalize(image_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        original_data = image_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image_data.shape)
        original_data = np.clip(original_data, 0., 1.)
    else:
        original_data = image_data

    outputs = None
    # get the output image of the convolutional autoencoder
    decoded = model.predict(original_data)

    for i in range(0, 10):
        # retrieve one original image and its recovered counterpart
    	original = (original_data[i] * 255).astype("uint8")
    	recovered = (decoded[i] * 255).astype("uint8")

    	# stack the original and reconstructed image side-by-side
    	output = np.hstack([original, recovered])

    	# if the outputs array is empty, initialize it as the current
    	if outputs is None:
    		outputs = output

    	# otherwise, vertically stack the outputs
    	else:
    		outputs = np.vstack([outputs, output])

    #save to file
    cv2.imwrite(name+".png",outputs)




################################################################################
#                                                                              #
#                                                                              #
#                   Here starts new function implementations                   #
#                                                                              #
################################################################################

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def visualize_class_activation_map(model, data):
        """
        CAM approach based on https://github.com/jacobgil/keras-cam/blob/master/cam.py
        """
        modeltype = 'cnn'
        if modeltype == 'cnn':
            print(data.shape)
            width, height, _ = data[0].shape
            class_weights = model.layers[-5].get_weights()
            final_conv_layer = get_output_layer(model, "conv2d_9")
            print(final_conv_layer)
            print(model.layers[0].input)
            get_output = K.function([model.layers[0].input], \
                        [final_conv_layer.output,
            model.layers[-5].output])
            [conv_outputs, predictions] = get_output([data])

            print(conv_outputs.shape)
        else:
            width, height, _ = data[0].shape
            auto, enc, pre = model.getModel()

            #Get the last layer of the encoder (global average pool layer) .
            class_weights = enc.layers[-1].get_weights()
            final_conv_layer = get_output_layer(enc, "encoded5")
            print(final_conv_layer)
            print(enc.layers[0].input)
            get_output = K.function([enc.layers[0].input], \
                        [final_conv_layer.output,
            enc.layers[-1].output])
            [conv_outputs, predictions] = get_output([data])

            print(conv_outputs.shape)

        outputs = None
        for i in range(0, 10):
            # retrieve one original image and its recovered counterpart
            original = (data[i] * 255).astype("uint8")

            conv_output = conv_outputs[i, :, :, :]
            prediction = predictions[i,:]

        	#Create the class activation map.
            cam = np.zeros(dtype = np.float32, shape = (28,28))

            #print(conv_output[:, :,0])
            print(conv_output.shape)
            print(prediction)
            for k in range(len(prediction)):
                cam += prediction[k] * conv_output[:, :,k]

            cam /= np.max(cam)
            print(cam)
            cam = cv2.resize(cam, (224, 224))

            #heatmap = cam

            #Depending on network output the color intensity making red highest is either of these:
            heatmap = cv2.applyColorMap((255*cam).astype("uint8"), cv2.COLORMAP_JET)
            #heatmap = cv2.applyColorMap(np.uint8(255 * (255 - heatmap)), cv2.COLORMAP_JET)
            heatmap[np.where(cam < 0.2)] = 0
            #add in original image
            recovered = heatmap*0.5 + original

        	# stack the original and reconstructed image side-by-side
            #output = np.hstack([original, recovered])

        	# if the outputs array is empty, initialize it as the current
            if outputs is None:
                outputs = recovered

        	# otherwise, vertically stack the output
            else:
                outputs = np.hstack([outputs, recovered])

        cv2.imwrite("output_path"+".png", outputs)



def visualize_activation_map(model, data):
    """
    Get the output activation maps from the network
    Based on https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    """

    width, height, _ = data[0].shape
    auto, enc, pre = model.getModel()

    #Choose which layers to get output from (Choose convolutional layers)
    ixs = [2,5,9,12,19,22,26,29,32]
    outputs = [enc.layers[i].output for i in ixs]
    model = Model(inputs=enc.inputs, outputs=outputs)

    # get feature map for first hidden layer
    x = np.expand_dims(data[3], axis=0)
    feature_maps = model.predict(x)
    # plot the output from each block
    square = 8
    j = 0
    #print(feature_maps.shape)
    for fmap in feature_maps:
        #print(fmap.shape)
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[0,:, :, ix-1], cmap='gray')
                ix += 1

        # show the figur
        plt.savefig(str(j)+'_image.png')
        j += 1


from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import median_filter
def kernel_inspection(model,data):
    """
    Test to make an optimal input image based on noise
    Based on: https://gist.github.com/RaphaelMeudec/31b7bba0b972ec6ec80ed131a59c5b3f#file-kernel_visualization-py
    !!! not working as well as expected -> might work better on more advanced images
    """

    # Layer name to inspect
    layer_name = 'conv2d_1'

    epochs = 100
    step_size = 0.3
    filter_index = 0
    # Create a connection between the input and the target layer
    auto, enc, pre = model.getModel()
    submodel = tf.keras.models.Model([enc.inputs[0]], [enc.get_layer(layer_name).output])
    vstack = None
    for i in range(2):
        hstack = None
        for j in range(5):
            # Initiate random noise
            input_img_data = np.random.random((1, 64, 64, 1))
            #input_img_data = (input_img_data - 0.5) * 20 + 128.
            input_img_data = median_filter(input_img_data, size=5)
            # Cast random noise from np.float64 to tf.float32 Variable
            input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

            # Iterate gradient ascents


            for _ in range(epochs):
                with tf.GradientTape() as tape:
                    outputs = submodel(input_img_data)
                    loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
                    grads = tape.gradient(loss_value, input_img_data)
                    normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
                    input_img_data.assign_add(normalized_grads * step_size)

                    a= input_img_data.numpy()
                    #blurred = gaussian_filter(a, sigma=1)
                    blurred = median_filter(a, size=5)
                    input_img_data = tf.Variable(tf.cast(blurred, tf.float32))

            filter_index += 1
            a= input_img_data.numpy()

            print(a.max())
            print(filter_index)
            print(input_img_data.shape)
            input = a[0]/a.max()
            print(input.shape)
            # if the outputs array is empty, initialize it as the current
            original = (input * 255).astype("uint8")
            if hstack is None:
                hstack = original
            # otherwise, vertically stack the output
            else:
                hstack = np.hstack([hstack, original])


        # if the outputs array is empty, initialize it as the current
        if vstack is None:
            vstack = hstack
        # otherwise, vertically stack the output
        else:
            vstack = np.vstack([vstack, hstack])


    cv2.imwrite("sdfd"+".png",vstack)
    tf.keras.preprocessing.image.save_img(
        "common.png", vstack
    )

def confusion_matrix(y,y_pred,save_name = "confusion_matrix.png"):
    """
    Makes a confusion matrix with heatmap
    Taken from: https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
    """

    import seaborn as sns
    import sklearn.metrics
    import matplotlib.pyplot as plt
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.savefig(save_name)


def accuracy(y,y_pred):
    """
    Calculates accuracy of an unsupervised algorithm using hungarian algorithm (linear assignment in sklearn)
    Taken from: https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
    """
    from sklearn.utils.linear_assignment_ import linear_assignment

    y_true = y.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(-w)

    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    print("ACC = ",acc)
    return acc

def visualize_class_predictions(x_data,y_data,y_pred,save_name ="visualize_class_predictions"):
    """
        This function plots 100 images with corresponding prediction for a visualization of
        the performance of the classifier for different looking class images.
        - Can help to interpret confusion_matrix results
        :param x_data: dataset
        :param y_data: dataset labels
        :param y_pred: dataset predictions
    """
    # choose a color palette with seaborn.
    num_classes = len(np.unique(y_pred))
    palette = np.array(sns.color_palette("hls", num_classes))
    sns.set_palette(palette)

    #sort classes from 0 to num classes
    j = 0

    for i in range(len(np.unique(y_data))):
        print(i)
        x_, y_, yp_ = getSameSamples(x_data,y_data,y_pred,i)
        img_numbers = min(15, len(x_))
        print(img_numbers)
        if i == 0:
            x = x_[:img_numbers,:,:]
            y = y_[:img_numbers]
            yp = yp_[:img_numbers]
        else:
            x = np.concatenate((x, x_[:img_numbers,:,:]), axis=0)
            y = np.concatenate((y, y_[:img_numbers]), axis=0)
            yp = np.concatenate((yp, yp_[:img_numbers]), axis=0)

    # make figure
    f = plt.figure(figsize=(16, 12))
    plt.title("Class predictions",fontsize = 20)

    width = 1000
    height = 1000
    max_dim = 100
    full_image = Image.new('RGB', (width, height), (255, 255, 255))
    data = x

    x = 0; y = 0
    for img, c in tqdm(zip(data, yp)):
        tile = Image.fromarray(np.uint8(img * 255)).convert('RGB')
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((max(1, int(tile.width / rs)), max(1, int(tile.height / rs))), Image.ANTIALIAS)

        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tile.size[0] - 1, tile.size[1] - 1], fill=None, outline=box_color_list[int(c)], width=5)

        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)))
        if x == 0.7:
            x = 0
            y =  0.1+y

        else:
            x = x + 0.1

    # Save full image
    plt.imshow(full_image)
    plt.savefig(save_name+'_image.png')
    plt.close()
    return

def getSameSamples(x_train,y_train,y_pred,class_name):

    #y_train = tf.keras.utils.to_categorical(y_train, num_classes=5, dtype='bool')
    print(y_train)
    train_mask = np.isin(y_train,[class_name])
    print(train_mask)
    x_train, y_train, y_pred= x_train[train_mask], y_train[train_mask], y_pred[train_mask]

    print(x_train.shape)
    return x_train, y_train, y_pred
