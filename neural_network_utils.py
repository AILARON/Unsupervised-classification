import time
import numpy as np
import os
import math

def saveWeights(model,name):
    # Save JSON config to disk
    json_config = model.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(json_config)
        # Save weights to disk
        print("[Info] saving weights")
        model.save_weights(str(name)+'_weights.h5')

def loadWeights(model,name):
    print("[Info] loading previous weights")
    try:
        model.load_weights(str(name)+'_weights.h5')
    except:
        print("Could not load weights")

    return model

def compute_features(dataset, model,N):

    print('Compute features')

    end = time.time()

    for i, (input_tensor) in enumerate(dataset):

        if i <= (N//32 - 1):
            #print('# lll: ',i)
            aux = model(input_tensor)
        else:
            #print('# DEBUG: ',i)
            input_tensor = input_tensor[0:N-i* 32]
            aux = model(input_tensor)

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        if i <= (N//32 - 1):

            features[i * 32: (i + 1) * 32] = aux
        else:

            features[i * 32:] = aux

        if i == math.ceil(N/32) - 1:
            break

    return features

def sortLabels(y_true,y_pred):
    print(y_pred.shape)
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(-w)

    new_pred = np.zeros(len(y_pred), dtype=np.int64)
    for i in range(len(y_pred)):
        new_pred[i] = ind[1][y_pred[i]]

    print(new_pred.shape)
    return new_pred
