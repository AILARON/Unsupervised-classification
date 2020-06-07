###############
#Regularizers for auto encoder models
###############

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K


############
#sparse activity regularizer -> Implements a kl_divergence loss function based on the explanation
#https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf

#code source:
#source https://github.com/mrquincle/keras-adversarial-autoencoders/blob/master/experiments/Sparse%20Ordinary%20Autoencoder.ipynb
#https://stackoverflow.com/questions/36913281/how-do-i-correctly-implement-a-custom-activity-regularizer-in-keras
############
class SparseActivityRegularizer(Regularizer):

    def __init__(self, p=0.1, sparsityBeta=3.0):
        self.p = K.cast_to_floatx(p)
        self.sparsityBeta = K.cast_to_floatx(sparsityBeta)

    def __call__(self, x):
        regularization = 0
        p_hat = K.mean(x, axis=0)
        regularization += self.sparsityBeta * K.sum(self.kl_divergence(self.p, p_hat))

        return regularization

    def kl_divergence(self, p, p_hat):
        return (p * K.log(p / p_hat)) + ((1-p) * K.log((1-p) / (1-p_hat)))

    def get_config(self):
        return {"name": self.__class__.__name__, "p": float(self.p), "beta": float(self.sparsityBeta)}
