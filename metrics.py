import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,accuracy_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

NMI = normalized_mutual_info_score
ARI = adjusted_rand_score

def acc(y_true, y_pred):
    y_pred = sortLabels(y_true, y_pred)
    return accuracy_score(y_true, y_pred)

def clustering_acc(y_true, y_pred):
    return acc(y_true, y_pred)

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
