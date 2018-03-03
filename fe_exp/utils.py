#conding=utf-8

import sys

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

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
    def __init__(self, method = 'l1', ratio = 0.1, data, labels):
        self.data = data
        self.labels = labels
        self.target_feature = int(data.shape[1] * ratio)
        self.excute(method)

    def excute(self, method):
        if method == 'l1':
            l1based()
        elif method == 'l2':
            l2based
        elif method == 'tree':
            treebased()
        elif method == 'MI':
            MIbased()
        elif method == 'deep':
            pass
        else:
            sys.exit('fs method not found!')

    def l1based(self):
        n_target = self.target_feature
        lr = LogisticRegression(C = 1, penalty = 'l1', tol = 0.1).fit(self.data, self.labels)
        coef = lr.coef_
        top_indices = np.argsort(-coef)[:,:n_target].ravel()
        data_new = data[:,top_indices]
        return data_new

    def l2based(self):
        n_target = self.target_feature
        lr = LogisticRegression(C = 1, penalty = 'l2', tol = 0.1).fit(self.data, self.labels)
        coef = lr.coef_
        top_indices = np.argsort(-coef)[:,:n_target].ravel()
        data_new = data[:,top_indices]
        return data_new

    def treebased(self):
        n_target = self.target_feature
        forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
        forest.fit(self.data, self.labels)
        importances = forest.feature_importances_
        top_indices = np.argsort(-importances)[:n_target].ravel()
        data_new = data[:,top_indices]
        return data_new

    def MIbased(self):
        n_target = self.target_feature
        data_new = SelectKBest(mutual_info_classif, k = n_target).fit_transform(self.data, self.labels)
        return data_new
    
    def deepbased(self):
        pass

class Classifier(object):
    def __init__(self, method = 'knn', data, labels):
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
        

