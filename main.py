import os
gpu = 1
if gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from auto_encoder import AutoEncoder
from load_dataset import LoadDataset
from clustering_algorithms import TSNEAlgo
from augment_data import DataAugmentation
from regularized_network_model import RegularizedNetworkModel
from network_model import NetworkModel
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from tensorflow.keras.models import Model
import os
from sklearn.cluster import SpectralClustering
from unet import UNET

from cbof import initialize_bof_layers

from utils import visualize_input_output,kernel_inspection,visualize_class_activation_map,confusion_matrix,visualize_class_predictions,accuracy

from bof_utils import getSameSamples, makeSamplevec, printVec

from clustering_algorithms import HierarchicalClustering, KMeansClustering

from results import getNetworkResults
#from variational_auto_encoder import VariationalAutoEncoder
#Import of tensorflow and setup gpu
import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_visible_devices(gpus[1:], 'GPU')
#print(tf.VERSION)

if gpu == 1:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')

    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    #tf.config.experimental.set_memory_growth(physical_devices[1], True)


def dataAugment():
    load_dir = "dataset/trainoriginalfull/"
    save_dir = "dataset/kaggle_augmented_train_new/"
    augment = DataAugmentation(load_dir,save_dir)
    augment.dataAugmentation()


def buildNetwork(model_type,latent_vector,latent_dim = 10, epochs = 10,train = True,noisy=False):
    # build and train regularized model
    if noisy:
        model = RegularizedNetworkModel("dataset/trainoriginalfull/",
        latent_vector = latent_vector,latent_dim = latent_dim, model_type=model_type, num_epochs = epochs)
    else:
        model = NetworkModel("dataset/trainoriginalfull/",
        latent_vector = latent_vector,latent_dim = latent_dim, model_type=model_type, num_epochs = epochs)
    #model = NetworkModel("dataset/kaggle_augmented_train",conv_layers = [256,128,64],
    #latent_vector = 32, model_type=model_type, num_epochs = 50)
    if train == True:
        model.runModel()
        model.saveWeights()
    else:
        model.buildModel()
        model.loadWeights()

    return model



def test(model,model_type):
    # get autoencoder, encoder and decoder
    auto, enc, pre = model.getModel()
    #Load test dataset
    test_data = LoadDataset("dataset/kaggle_original_test/",0)
    test_data, test_label, val, val_label = test_data.load_data()
    visualize_input_output(auto,test_data,model_type)


def make_confusion_kmean():
    y_pred  = np.zeros(3720)
    y_true = np.zeros(3720)
    for i in range(4500):
        if i <= 506:
            y_true[i] = 0
            y_pred[i] = 4
        elif i == 507:
            y_true[i] = 0
            y_pred[i] = 2
        elif i == 508:
            y_true[i] = 0
            y_pred[i] = 3
    j = 0
    for i in range(509,1180):
        if j < 330:
            y_true[i] = 1
            y_pred[i] = 1
        elif j < 330 + 7:
            y_true[i] = 1
            y_pred[i] = 2
        elif j < 330 + 7+ 20:
            y_true[i] = 1
            y_pred[i] = 4
        elif j < 330 + 7 + 20 + 313:
            y_true[i] = 1
            y_pred[i] = 0
        elif j < 330 + 7 + 20 + 313+1:
            y_true[i] = 1
            y_pred[i] = 3
        j += 1

    j = 0
    for i in range(1180,1681):
        if j < 90:
            y_true[i] = 2
            y_pred[i] = 1
        elif j < 90 + 199:
            y_true[i] = 2
            y_pred[i] = 2
        elif j < 90 + 199 + 117:
            y_true[i] = 2
            y_pred[i] = 4
        elif j < 90 + 199 + 117 +89 :
            y_true[i] = 2
            y_pred[i] = 0
        elif j < 90 + 199 + 117 +89 + 6:
            y_true[i] = 2
            y_pred[i] = 3
        j += 1
    j = 0
    for i in range(1681,2843):
        if j < 112:
            y_true[i] = 3
            y_pred[i] = 1
        elif j < 112 + 278:
            y_true[i] = 3
            y_pred[i] = 2
        elif j < 112 + 278 +97:
            y_true[i] = 3
            y_pred[i] = 4
        elif j < 112 + 278 +97 +64 :
            y_true[i] = 3
            y_pred[i] = 0
        elif j < 112 + 278 +97 +64 + 611:
            y_true[i] = 3
            y_pred[i] = 3
        j += 1
    j = 0
    for i in range(2843,3720):
        if j < 46:
            y_true[i] = 4
            y_pred[i] = 1
        elif j < 46 + 7:
            y_true[i] = 4
            y_pred[i] = 2
        elif j < 46 + 7 +708:
            y_true[i] = 4
            y_pred[i] = 4
        elif j < 46 + 7 +708 +115 :
            y_true[i] = 4
            y_pred[i] = 0
        elif j < 112 + 7 +708 +115 + 1:
            y_true[i] = 4
            y_pred[i] = 3
        j += 1

    confusion_matrix(y_true,y_pred, save_name = "testlitest.png")

def make_confusion_SC():
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


from test import vae_test



if __name__=="__main__":
    tf.keras.backend.clear_session()

    #train_data = LoadDataset("dataset/kaggle_original_train/",0)
    #train_data, train_label, val, val_label = train_data.load_data()

    #make_confusion_kmean()
    #make_confusion_SC()


    model = "vae"
    if model == "vae":
        train_data = LoadDataset("dataset/trainoriginalfull/",0.1)
        train_data, test_label, val_data, val_label = train_data.load_data()
        from variational_auto_encoder import vae

        model = vae(train_data,val_data)

        train_data = LoadDataset("dataset/kaggle_original_train/",0)
        train_data, train_label, val, val_label = train_data.load_data()
        x  = train_data.reshape(train_data.shape[0], 64, 64, 1).astype('float32')
        #x = x[:300,:,:,:]
        print(x.shape)

        ds = np.zeros((train_data.shape[0],64))
        i = 0
        train_dataset = tf.data.Dataset.from_tensor_slices(x).batch(32)
        for train_x in train_dataset:
            data,_ = model.encode(train_x)
            for enc in data:
                ds[i] = enc
                i = i +1

        TSNE = TSNEAlgo()
        TSNE.tsne_fit(ds,perplexity = 35)

        kmean = KMeansClustering()
        kmean.fit(ds)
        predKM  = kmean.predict(ds)
        accuracy(train_label,predKM)



        model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',
                                   assign_labels='kmeans')
        predSPEC = model.fit_predict(ds)
        accuracy(train_label,predSPEC)

        train_x = np.reshape(train_data,(3720,64,64))
        TSNE.tsne_plot(train_x,predSPEC,save_data_dir ="vae",save_name="spectral")
        TSNE.tsne_plot(train_x,train_label,save_data_dir ="vae",save_name="true_label")
        TSNE.tsne_plot(train_x,predKM,save_data_dir ="vae",save_name="kmean")



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


    #test_vae()
    test_model = False
    if test_model == True:
        model_type = "COAPNET"
        latent_vector = "globalAverage"
        model = buildNetwork(model_type, latent_vector,latent_dim = 64, epochs = 50,train = False,noisy = False)

        auto, enc, pre = model.getModel()

        #enc_output = enc.predict(train_data)
        #kmean = KMeansClustering()
        #kmean.fit(enc_output)
        #pred  = kmean.predict(enc_output)
        #accuracy(train_label,pred)
        getNetworkResults(model,latent_vector)

    test_DEC = False
    if test_DEC == True:
        train_data = LoadDataset("dataset/kaggle_original_train/",0)
        train_data, train_label, val, val_label = train_data.load_data()
        encoded_x= np.reshape(train_data,(3720, 64 *64))

        model_type = "COAPNET"
        latent_vector = "globalAverage"
        model = buildNetwork(model_type, latent_vector,latent_dim = 64, epochs = 5,train = True,noisy = False)

        auto, enc, pre = model.getModel()


        from DEC import DeepEmbeddedClustering
        from results import getDECNetworkResults

        dec = DeepEmbeddedClustering(auto,enc,train_data,train_label,5)
        dec.buildModel()
        dec.trainModel()
        dec,enc = dec.getModel()
        getDECNetworkResults(dec,enc)


    ############################################################################

    """############################ DEC ###################################
    from DEC import ClusteringLayer
    from cbof import BoF_Pooling
    from tensorflow.keras.layers import Dense
    #model_type = "fullyConnected"
    #model = buildNetwork(model_type,train = False)
    #auto, enc, pre = model.getModel()
    n_clusters = 5
    pooling = BoF_Pooling(128,spatial_level=1, name="BOF")(enc.get_layer("conv2d_8").output)
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(pooling)
    model = Model(inputs=enc.input,
                       outputs=[clustering_layer, auto.output])
    #initialize_bof_layers(model,train_data)
    #from tensorflow.keras.utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes=True)
    #from IPython.display import Image
    #Image(filename='model.png')
    if model_type == "fullyConnected":
        x = np.reshape(train_data,(3720, 64 *64))
        y =train_label
    else:
        x = train_data
        y =train_label
    import sklearn.metrics
    from keras import metrics
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    pretrain_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.01, nesterov=False)
    #pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
     #.reshape((test_data.shape[0], -1))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    #y_pred = kmeans.fit_predict(enc.predict(x))
    #model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    y_pred_last = 0#np.copy(y_pred)
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=pretrain_optimizer)
    loss = 0
    index = 0
    maxiter = 20000
    update_interval = 140
    index_array = np.arange(train_data.shape[0])
    tol = 0.000001 # tolerance threshold to stop training
    batch_size = 32
    import metrics
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q, _  = model.predict(x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p
            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y is not None:
                acc = np.round(metrics.acc(y, y_pred), 5)
                nmi = np.round(metrics.nmi(y, y_pred), 5)
                ari = np.round(metrics.ari(y, y_pred), 5)
                loss = np.round(loss, 5)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc,nmi, ari), ' ; loss=', loss)
            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    # Eval.
    q, _ = model.predict(x, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p
    encoded_train = enc.predict(x)
    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if y is not None:
        acc = np.round(metrics.acc(y, y_pred), 5)
        nmi = np.round(metrics.nmi(y, y_pred), 5)
        ari = np.round(metrics.ari(y, y_pred), 5)
        loss = np.round(loss, 5)
        print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)
    confusion_matrix(train_label.astype(np.int64),y_pred)
    TSNE = TSNEAlgo
    train_x = np.reshape(x,(3720,64,64))
    TSNE.tsne_fit(encoded_train,train_x,y_pred.astype(int),save_name="Pred",
        save_data_dir = "spectral",perplexity = 7*5)
    TSNE.tsne_fit(encoded_train,train_x,train_label.astype(int),save_name="True",
        save_data_dir = "spectral",perplexity = 7*5)
    #################################### HDBSCAN ##########################
    accold =0
    index = 0
    for i in range(2,100):
        encoder = hdbscan.HDBSCAN(min_cluster_size=i)
        cluster_labels = encoder.fit_predict(encoded_train)
        acc = accuracy(train_label,cluster_labels)
        if acc > accold:
            accold = acc
            index = i
    print(accold,i)
    """
    """train_data = LoadDataset("dataset/kaggle_original_test/",0)
    train_data, train_label, val, val_label = train_data.load_data()
    auto, enc, pre = model.getModel()
    class_1, class_1_label, =  getSameSamples(train_data,train_label,1.0)
    pred = enc.predict(class_1)
    samplevec_1 = makeSamplevec(pred)
    class_1, class_1_label, =  getSameSamples(train_data,train_label,2.0)
    pred = enc.predict(class_1)
    samplevec_2 = makeSamplevec(pred)
    class_1, class_1_label, =  getSameSamples(train_data,train_label,3.0)
    pred = enc.predict(class_1)
    samplevec_3 = makeSamplevec(pred)
    printVec(samplevec_1,samplevec_2,samplevec_3)"""

    #Create new data using the DataAugmentation class:
    #imagedata()

    #Load train dataset
    #train_data = LoadDataset("dataset/kaggle_augmented_train/",0.1)
    #train_data, train_label, val, val_label = train_data.load_data()

    #Load test dataset
    #test_data = LoadDataset("dataset/kaggle_original_test/",0)
    #test_data, test_label, val, val_label = test_data.load_data()

    """model = UNET()
    auto = model.build_unet()
    stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True)
    history = auto.fit(x = train_data,y = train_data,batch_size = 32,
    validation_data = (val,val), callbacks = [stopping],shuffle=True,
    epochs = 50,verbose= 1)
    #Load test dataset
    test_data = LoadDataset("dataset/kaggle_original_test/",0)
    test_data, test_label, val, val_label = test_data.load_data()
    visualize_input_output(auto,test_data,"test")"""
    #Load test dataset
    """train_data = LoadDataset("dataset/kaggle_original_test/",0)
    train_data, train_label, val, val_label = train_data.load_data()
    #x_train, y_train = getSameSamples(train_data,train_label,1)
    #Create and build new model
    model = trainNetwork(model_type)
    # get autoencoder, encoder and decoder
    auto, enc, pre = model.getModel()
    #hierarchical CLUSTERING
    # Hierarchical Clustering
    imgHeight = 64
    imgWidth = 64
    imgNumber = 50
    #something something
    encoded_x = enc.predict(train_data[:imgNumber,:])
    labels = HierarchicalClustering()
    average = labels.draw_dendogram(encoded_x,title='Hierarchical Clustering Dendrogram')
    train_y = average[:imgNumber]
    train_x = np.reshape(train_data[:imgNumber,:],(imgNumber,imgHeight,imgWidth))
    print("shapes = ", encoded_x.shape,train_x.shape,train_y.shape)
    TSNE = TSNEAlgo
    for i in range (10):
        TSNE.tsne_fit(encoded_x,train_x,train_y.astype(int),save_name=str(i*5),
            save_data_dir = "test",perplexity = i*5)"""
    """# Plot latent vector
    class_1, class_1_label, =  getSameSamples(train_data,train_label,1.0)
    pred = enc.predict(class_1)
    samplevec_1 = makeSamplevec(pred)
    class_1, class_1_label, =  getSameSamples(train_data,train_label,2.0)
    pred = enc.predict(class_1)
    samplevec_2 = makeSamplevec(pred)
    class_1, class_1_label, =  getSameSamples(train_data,train_label,3.0)
    pred = enc.predict(class_1)
    samplevec_3 = makeSamplevec(pred)
    printVec(samplevec_1,samplevec_2,samplevec_3)"""

    """
    #Predict results using TSNE
    tsne(enc, train_data,train_label,model_type,model_type)
    # Visualize input and output from autoencoder
    visualize_input_output(auto,train_data,model_type)
    """
