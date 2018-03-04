#conding=utf-8

import sys

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

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
            knnclassifier()
        else:
            sys.exit('method not fount!')

    def lrclassifier(self):
        pass

    def svmclassifier(self):
        pass

    def knnclassifier(self):
        data = self.data
        labels = self.labels
        kf_data, test_data, kf_labels, test_labels = train_test_split(data, labels, test_size = 0.25, random_state = 0)
        kf = KFold(n_splits = 5, shuffle = False, random_state = 0)
        models = []
        preformance = []
        for kf_train_index, kf_test_index in kf.split(kf_data):
            kf_train_data, kf_test_data = kf_data[kf_train_index], kf_data[kf_test_index]
            kf_train_labels, kf_test_labels = kf_labels[kf_train_index], kf_labels[kf_test_index]
            knn = KNeighborsClassifier(n_neighbors = 3)
            knn.fit(kf_train_data, kf_train_labels)
            pred_labels = knn.predict(kf_test_data)
            acc = accuracy_score(kf_test_labels, pred_labels)
            models.append(knn)
            performance.append(acc)

        top3_indices = np.argsort(-np.array(performance)).ravel()[:3]
        preds = []
        for model in [models[i] for i in top3_indices]:
            cur_pred = model.predict(test_data)
            preds.append(cur_pred)

        pred = vote(preds)
        report(test_labels, pred)

    def vote(self, preds):
        pred = np.sum(np.vstack((preds[0], preds[1], preds[2])), axis = 0)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred
    
    def report(self, labels, pred):
        acc = accuracy_score(labels, pred)
        prfs = classification_report(labels, pred)
        print 'accuracy: {0:.4f}'.format(acc)
        print prfs

