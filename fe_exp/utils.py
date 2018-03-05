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

def load_data(datapath, labelpath):
    data = np.loadtxt(open(datapath, 'rb'))
    labels = np.loadtxt(open(labelpath, 'rb'))
    print "raw dimension: {0}".format(data.shape)
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
    def __init__(self, method, ratio, data, labels):
        self.data = data
        self.labels = labels
        self.method = method
        self.target_feature = int(data.shape[1] * ratio)
        self.excute()

    def excute(self):
        method = self.method
        if method == 'base':
            return self.base_line()
        elif method == 'l1':
            return self.l1based()
        elif method == 'l2':
            return self.l2based()
        elif method == 'tree':
            return self.treebased()
        elif method == 'MI':
            return self.MIbased()
        elif method == 'deep':
            pass
        else:
            sys.exit('fs method not found!')

    def base_line(self):
        kf_data, test_data, kf_labels, test_labels = self.split_data()
        return kf_data, test_data, kf_labels, test_labels

    def l1based(self):
        n_target = self.target_feature
        kf_data, test_data, kf_labels, test_labels = self.split_data()
        lr = LogisticRegression(C = 1, penalty = 'l1', tol = 0.1).fit(kf_data, kf_labels)
        coef = lr.coef_
        top_indices = np.argsort(-coef)[:,:n_target].ravel()
        kf_data_new = kf_data[:,top_indices]
        test_data_new = test_data[:,top_indices]
        return kf_data_new, test_data_new, kf_labels, test_labels

    def l2based(self):
        n_target = self.target_feature
        kf_data, test_data, kf_labels, test_labels = self.split_data()
        lr = LogisticRegression(C = 1, penalty = 'l2', tol = 0.1).fit(kf_data, kf_labels)
        coef = lr.coef_
        top_indices = np.argsort(-coef)[:,:n_target].ravel()
        kf_data_new = kf_data[:,top_indices]
        test_data_new = test_data[:,top_indices]
        return kf_data_new, test_data_new, kf_labels, test_labels

    def treebased(self):
        n_target = self.target_feature
        kf_data, test_data, kf_labels, test_labels = self.split_data()
        forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
        forest.fit(kf_data, kf_labels)
        importances = forest.feature_importances_
        top_indices = np.argsort(-importances)[:n_target].ravel()
        kf_data_new = kf_data[:,top_indices]
        test_data_new = test_data[:,top_indices]
        return kf_data_new, test_data_new, kf_labels, test_labels

    def MIbased(self):
        n_target = self.target_feature
        kf_data, test_data, kf_labels, test_labels = self.split_data()
        mi_select = SelectKBest(mutual_info_classif, k = n_target)
        kf_data_new = mi_select.fit_transform(kf_data, kf_labels)
        test_data_new = mi_select.transform(test_data)
        return kf_data_new, test_data_new, kf_labels, test_labels
    
    def deepbased(self):
        pass

    def split_data(self):
        kf_data, test_data, kf_labels, test_labels = train_test_split(self.data, self.labels, test_size = 0.25, random_state = 0)
        return kf_data, test_data, kf_labels, test_labels

class Classifier(object):
    def __init__(self, method, kf_data, test_data, kf_labels, test_labels):
        self.kf_data = kf_data
        self.test_data = test_data
        self.kf_labels = kf_labels
        self.test_labels = test_labels
        self.method = method
        self.excute()

    def excute(self):
        method = self.method
        if method == 'lr':
            pass
        elif method == 'svm':
            pass
        elif method == 'knn':
            self.knnclassifier()
        else:
            sys.exit('method not fount!')

    def lrclassifier(self):
        pass

    def svmclassifier(self):
        pass

    def knnclassifier(self):
        kf_data = self.kf_data
        test_data = self.test_data
        kf_labels = self.kf_labels
        test_labels = self.test_labels
        kf = KFold(n_splits = 5, shuffle = False, random_state = 0)
        models = []
        performance = []
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

        pred = self.vote(preds)
        self.report(test_labels, pred)

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
