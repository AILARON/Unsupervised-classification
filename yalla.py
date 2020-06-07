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

"""
Take the memory size of different models
"""
alloc = False
if alloc == True:
    pid = None
    import psutil

    process = psutil.Process(os.getpid())
    #print(process.memory_info().rss)  # in bytes
    train_data = LoadDataset("dataset/kaggle_original_train/",0)
    train_data, train_label, val, val_label = train_data.load_data()


    #tracemalloc.start()

    #test_vae()
    model_type = "COAPNET"
    latent_vector = "globalAverage"
    model = buildNetwork(model_type, latent_vector,latent_dim = 64, epochs = 1,train = False,noisy = False)


    auto, enc, pre = model.getModel()
    #train_data= np.reshape(train_data,newshape  = (train_data.shape[0],64*64*1))
    enc_output = enc.predict(train_data)
    print(process.memory_info())  # in bytes

def make_confusion_SC():
    """
    Fix confusion matrix so that true and predicted class both have same label
    """
    y_pred  = np.zeros(3720)
    y_true = np.zeros(3720)
    for i in range(3720):
        if i < 4:
            y_true[i] = 0
            y_pred[i] = 4
        if i < 4 + 505:
            y_true[i] = 0
            y_pred[i] = 0
    j = 0
    for i in range(509,1180):
        if j < 203:
            y_true[i] = 1
            y_pred[i] = 4
        elif j < 203 + 3:
            y_true[i] = 1
            y_pred[i] = 0
        elif j < 203 + 3+ 4:
            y_true[i] = 1
            y_pred[i] = 3
        elif j < 203 + 3+ 4 + 461:
            y_true[i] = 1
            y_pred[i] = 1
        j += 1

    j = 0
    for i in range(1180,1681):
        if j < 87:
            y_true[i] = 2
            y_pred[i] = 4
        elif j < 87 + 185:
            y_true[i] = 2
            y_pred[i] = 0
        elif j < 87 + 185 +75:
            y_true[i] = 2
            y_pred[i] = 3
        elif j < 87 + 185 +75 +28 :
            y_true[i] = 2
            y_pred[i] = 1
        elif j < 87 + 185 +75 +28 + 126:
            y_true[i] = 2
            y_pred[i] = 2
        j += 1
    j = 0
    for i in range(1681,2843):
        if j < 123:
            y_true[i] = 3
            y_pred[i] = 4
        elif j < 123 + 37:
            y_true[i] = 3
            y_pred[i] = 0
        elif j < 112 + 37 +995:
            y_true[i] = 3
            y_pred[i] = 3
        elif j < 112 + 37 +995 +7 :
            y_true[i] = 3
            y_pred[i] = 1
        j += 1
    j = 0
    for i in range(2843,3720):
        if j < 828:
            y_true[i] = 4
            y_pred[i] = 4
        elif j < 828 + 43:
            y_true[i] = 4
            y_pred[i] = 0
        elif j < 828 + 43 +3 :
            y_true[i] = 4
            y_pred[i] = 3
        elif j < 828 + 43 +3 +3  :
            y_true[i] = 4
            y_pred[i] = 1

        j += 1

    confusion_matrix(y_true,y_pred, save_name = "testlitestSC.png")
