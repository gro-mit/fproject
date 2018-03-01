#conding=utf-8

from sklearn import preprocessing
import numpy as np
import pandas as pd

class PreProcessing(object):
    def __init__(self, data):
        self.data = data

    def imputation(missing_values, strategy = 'mean'):
        imp = preprocessing.Imputer(missing_values, strategy)
        self.data = imp.fit()
    def standarization():

def load_data(filepath):
    data = np.

