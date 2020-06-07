from bof_utils import getSameSamples, makeSamplevec, printVec
import cv2
import numpy as np

def printClasses(x,y):
    noise_factor = 0.05
    #x = cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    test_data = x
    test_label = y
    #test_data = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image_data.shape)
    #test_data = np.clip(test_data, 0., 1.)


    class_0, class_1_label, =  getSameSamples(test_data,test_label,0.0)
    class_1, class_1_label, =  getSameSamples(test_data,test_label,1.0)
    class_2, class_1_label, =  getSameSamples(test_data,test_label,2.0)
    class_3, class_1_label, =  getSameSamples(test_data,test_label,3.0)
    class_4, class_1_label, =  getSameSamples(test_data,test_label,4.0)

    outputs = (class_0[3] * 255).astype("uint8")
    output = (class_1[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])
    output = (class_2[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])
    output = (class_3[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])
    output = (class_4[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])


    cv2.imwrite("image_of_all"+".png", outputs)

    test_data = x
    test_label = y
    test_data = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    test_data = np.clip(test_data, 0., 1.)


    class_0, class_1_label, =  getSameSamples(test_data,test_label,0.0)
    class_1, class_1_label, =  getSameSamples(test_data,test_label,1.0)
    class_2, class_1_label, =  getSameSamples(test_data,test_label,2.0)
    class_3, class_1_label, =  getSameSamples(test_data,test_label,3.0)
    class_4, class_1_label, =  getSameSamples(test_data,test_label,4.0)

    outputs = (class_0[3] * 255).astype("uint8")
    output = (class_1[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])
    output = (class_2[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])
    output = (class_3[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])
    output = (class_4[0] * 255).astype("uint8")
    outputs = np.hstack([outputs, output])


    cv2.imwrite("image_of_all_noisy"+".png", outputs)
