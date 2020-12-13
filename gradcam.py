## https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()



    def find_target_layer(self):
            # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applie
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the mode
        gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output,
				self.model.output])
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
		# use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        print(heatmap)
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_JET):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid imag
        return (heatmap, output)

    # import the necessary packages
#from pyimagesearch.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
#import argparse
#import imutils
import cv2





def gradImage(model,image):
    orig = image.copy()

    # Preprocess image
    # Subtract mean and make it into -1 - 1 domain
    image = (image - 239.64)/255
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    print('Data min=%.3f, max=%.3f' % (image.min(), image.max()))

    # use the network to make predictions on the input image and find
    # the class label index with the largest corresponding probability
    preds = model.predict(image)
    i = np.argmax(preds[0])
    print(i)
    print(preds[0][i])

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (128, 128))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    return output

def printOutput(data):
    output = "3x3"
    if output == "3x3":
        hstack = None
        vstack = None
        for i in range (3):
            hstack = None
            for j in range (3):
                print(i*3 +j)
                original = data[i*3 +j]
                if hstack is None:
                    hstack = original
                else:
                    hstack = np.hstack([hstack, original])
            if vstack is None:
                vstack = hstack
            else:
                vstack = np.vstack([vstack, hstack])
        cv2.imwrite("output_gradcam.png",vstack)

from deep_neural_networks import VGG_BATCHNORM, RESNET101,RESNET50, COAPNET, RESNET, BOF_MODELS
from load_dataset import importWHOI, importKaggle, importKaggleOld

from tensorflow.keras import layers

def make_discriminator_model():
    model = tf.keras.Sequential( name='discriminator')
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1))

    return model

def grad():
    model = make_discriminator_model()
    model = loadWeights(model,"discriminator")

    train_data, _ = importKaggleOld(train = True)
    import cv2

    list = [16403,4953,29872,7892,23613,1747,19272,17585,21178]
    img = []
    #for i in range(16):
    #    list.append(randint(0,30000))
    for i, val in enumerate(list):
        img.append(train_data[val])


    data = np.array([cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_LINEAR) for img in (img)])

    images = []
    for i in range(9):
        orig = data[i].copy()
        orig=np.stack([orig]*3, axis=-1)
        orig = orig.reshape(128,128,3)
        orig = orig.astype("uint8")

        images.append(gradImage(model,orig))

    printOutput(images)
    #cv2.imwrite("output_gradcam"+".png", output)

    """

    image = orig.copy()

    image = (image - 239.64)/255
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    print('Data min=%.3f, max=%.3f' % (image.min(), image.max()))
    #data = data[0:32]
    #data=np.stack([data]*3, axis=-1)
    #data = data.reshape(-1, 224,224,3)




    # use the network to make predictions on the input image and find
    # the class label index with the largest corresponding probability
    preds = model.predict(image)
    i = np.argmax(preds[0])
    print(i)
    print(preds[0][i])
    # decode the ImageNet predictions to obtain the human-readable label
    #decoded = imagenet_utils.decode_predictions(preds)
    #(imagenetID, label, prob) = decoded[0][0]
    #label = "{}: {:.2f}%".format(label, prob * 100)
    #print("[INFO] {}".format(label))

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (224, 224))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # draw the predicted label on the output image
    #cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    #cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	#0.8, (255, 255, 255), 2)
    # display the original image and resulting heatmap and output image
    # to our screen
    #output = np.vstack([orig, heatmap, output])
    #output = imutils.resize(output, height=700)
    cv2.imwrite("output_gradcam"+".png", output)
    #cv2.waitKey(0)
    """
def loadWeights(model,name):
    print("[Info] loading previous weights")
    try:
        model.load_weights(name+'_weights.h5')
    except:
        print("Could not load weights")

    return model
