#conding=utf-8

import sys
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def load_data(datapath, labelpath):
    data = np.loadtxt(open(datapath, 'rb'))
    labels = np.loadtxt(open(labelpath, 'rb'))
    print "dimension: {0}".format(data.shape)
    return data, labels

class PreProcessing(object):
    def __init__(self, data):
        self.data = data

    def imputation(self, missing_values, strategy = 'mean'):
        imp = preprocessing.Imputer(missing_values, strategy, axis = 0)
        return imp.fit_transform(self.data)

    def standarization(self):
        return preprocessing.scale(self.data)

class FeatureSelection(object):
    def __init__(self, method = 'l1', data, labels):


class Classifier(object):
    def __init__(self, method = 'lr', data, labels):
        self.data = data
        self.labels = labels
        self.excute(method)

    def excute(self, method):
        if method == 'lr':
            pass
        elif method == 'svm':
            pass
        elif method == 'knn':
            pass
        else:
            sys.exit('method not fount!')

    def lrclassifier(self):
        pass

    def svmclassifier(self):
        pass

    def knnclassifier(self):
        

