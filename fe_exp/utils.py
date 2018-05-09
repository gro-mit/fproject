#conding=utf-8

import sys
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn import decomposition
from sklearn import random_projection
from sklearn import manifold

from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

from sklearn import cluster

import scipy.spatial
import scipy.cluster

#from dfs import DeepFeaturnSelection

import warnings
import sklearn.exceptions
#ignore 0 precision or recall warnings
warnings.filterwarnings('ignore', category = sklearn.exceptions.UndefinedMetricWarning)

def load_data(datapath, labelpath):
    data = np.loadtxt(open(datapath, 'rb'))
    labels = np.loadtxt(open(labelpath, 'rb'))
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
    def __init__(self, method, ratio, random_state, data, labels):
        self.data = data
        self.labels = labels
        self.method = method
        self.target_feature = int(data.shape[1] * ratio)
        self.random_state = random_state

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
        elif method == 'ftest':
            return self.fclassifbased()
        elif method == 'deep':
            return self.deepbased()
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
        forest = ExtraTreesClassifier(n_estimators = 250)
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
    
    def fclassifbased(self):
        n_target = self.target_feature
        kf_data, test_data, kf_labels, test_labels = self.split_data()
        ftest_select = SelectKBest(f_classif, k = n_target)
        kf_data_new = ftest_select.fit_transform(kf_data, kf_labels)
        test_data_new = ftest_select.transform(test_data)
        return kf_data_new, test_data_new, kf_labels, test_labels

    def deepbased(self):
        n_target = self.target_feature
        kf_data, test_data, kf_labels, test_labels = self.split_data()
        dfs = DeepFeaturnSelection(learning_rate = 0.01, batch_size = 50,\
                n_epochs = 10)
        dfs.fit(kf_data, kf_labels)
        importances = dfs.importances_
        top_indices = np.argsort(-importances)[:, :n_target].ravel()
        kf_data_new = kf_data[:, top_indices]
        test_data_new = test_data[:, top_indices]
        return kf_data_new, test_data_new, kf_labels, test_labels

    def split_data(self):
        kf_data, test_data, kf_labels, test_labels = train_test_split(self.data, self.labels, test_size = 0.25, random_state = self.random_state)
        return kf_data, test_data, kf_labels, test_labels

class Dimensionality_reduction(object):
    def __init__(self, method, ratio, kf_data, test_data, kf_labels, test_labels):
        self.method = method
        self.ratio = ratio
        self.kf_data = kf_data
        self.test_data = test_data
        self.kf_labels = kf_labels
        self.test_labels = test_labels
        self.n_target = int(kf_data.shape[1] * ratio)

    def excute(self):
        method = self.method
        if method == 'base' or self.ratio == 1:
            return self.base_line()
        elif method == 'pca':
            return self.pca()
        elif method == 'rp':
            return self.rp()
        elif method == 'mds':
            return self.mds()
        else:
            sys.exit('dr method not found!')

    def base_line(self):
        return self.kf_data, self.test_data, self.kf_labels, self.test_labels

    def pca(self):
        n_target = self.n_target
        if n_target >= self.kf_data.shape[1]:
            return self.kf_data, self.test_data, self.kf_labels, self.test_labels
        if n_target > self.kf_data.shape[0]:
            n_target = int((self.kf_data.shape[0] + self.test_data.shape[0]) * (0.25 + self.ratio))
        pca = decomposition.PCA(n_components = n_target)
        high_data = np.concatenate((self.kf_data, self.test_data), 0) if self.test_data.any() else self.kf_data
        low_data = pca.fit(high_data).transform(high_data)
        kf_data_new = low_data[0:self.kf_data.shape[0]]
        if self.test_data.any():
            test_data_new = low_data[self.kf_data.shape[0]:(self.kf_data.shape[0] + self.test_data.shape[0])]

        return kf_data_new, test_data_new, self.kf_labels, self.test_labels
            
    def rp(self):
        n_target = self.n_target
        if n_target >= self.kf_data.shape[1]:
            return self.kf_data, self.test_data, self.kf_labels, self.test_labels
        rp = random_projection.SparseRandomProjection(n_components = n_target)
        high_data = np.concatenate((self.kf_data, self.test_data), 0) if self.test_data.any() else self.kf_data
        low_data = rp.fit_transform(high_data)
        kf_data_new = low_data[0:self.kf_data.shape[0]]
        if self.test_data.any():
            test_data_new = low_data[self.kf_data.shape[0]:(self.kf_data.shape[0] + self.test_data.shape[0])]

        return kf_data_new, test_data_new, self.kf_labels, self.test_labels

    def mds(self):
        n_target = self.n_target
        n_init = 3
        max_iter = 300
        if n_target >= self.kf_data.shape[1]:
            return self.kf_data, self.test_data, self.kf_labels, self.test_labels

        high_data = np.concatenate((self.kf_data, self.test_data), axis = 0) if self.test_data.any() else self.kf_data
        low_data = manifold.MDS(n_components = n_target, max_iter = max_iter, n_init = n_init).fit_transform(high_data)
        kf_data_new = low_data[0:self.kf_data.shape[0]]
        if self.test_data.any():
            test_data_new = low_data[self.kf_data.shape[0]:(self.kf_data.shape[0] + self.test_data.shape[0])]
        
        return kf_data_new, test_data_new, self.kf_labels, self.test_labels


class Classifier(object):
    def __init__(self, method, kf_data, test_data, kf_labels, test_labels):
        self.kf_data = kf_data
        self.test_data = test_data
        self.kf_labels = kf_labels
        self.test_labels = test_labels
        self.method = method

    def excute(self):
        method = self.method
        if method == 'lr':
            pass
        elif method == 'svm':
            pass
        elif method == 'knn':
            return self.knnclassifier()
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
        kf = KFold(n_splits = 5, shuffle = False)
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
        acc, rpt = self.report(test_labels, pred)
        return acc, rpt

    def vote(self, preds):
        pred = np.sum(np.vstack((preds[0], preds[1], preds[2])), axis = 0)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred
    
    def report(self, labels, pred):
        acc = accuracy_score(labels, pred)
        prfs = classification_report(labels, pred)
        rpt = np.array(precision_recall_fscore_support(labels, pred)).T
        print 'acc: {0:.4f}'.format(acc)
        print prfs
        return acc, rpt

class Clustering(object):
    def __init__(self, method, measure, n_clusters, data, labels):
        self.method = method
        self.measure = measure
        self.n_clusters = n_clusters
        self.data = data
        labels[labels < 0] = 0
        self.labels = labels

        self.excute()

    def excute(self):
        method = self.method
        measure = self.measure

        if method == 'hc':
            return self.hc()
        elif method == 'kmeans':
            return self.kmeans()
        elif method == 'kmeans++':
            return self.kmeansplus()
        elif method == 'minibatchkmeans':
            return self.minibatchkmeans()
        else:
            sys.exit('method not fount!')

    def hc(self):
        #scipy measure: correlation euclidean mahalanobis cosine
        measure = 'correlation' if self.measure == 'pearson' else self.measure
        distMat = scipy.spatial.distance.pdist(self.data, measure)
        hc = scipy.cluster.hierarchy.linkage(distMat, method = 'average')
        clusters = scipy.cluster.hierarchy.fcluster(hc, self.n_clusters, criterion = 'maxclust')
        pred = clusters - 1.0
        return self.report(self.labels, pred), pred

    def kmeans(self):
        kmeans_cluster = cluster.KMeans(n_clusters = self.n_clusters, init = 'random')
        kmeans_cluster.fit(self.data)

        return self.report(self.labels, kmeans_cluster.labels_), kmeans_cluster.labels_

    def kmeansplus(self):
        kmeans_cluster = cluster.KMeans(n_clusters = self.n_clusters, init = 'k-means++')
        kmeans_cluster.fit(self.data)

        return self.report(self.labels, kmeans_cluster.labels_), kmeans_cluster.labels_

    def minibatchkmeans(self):
        minibatch_kmeans = cluster.MiniBatchKMeans(n_clusters = self.n_clusters, init = 'k-means++', batch_size = 50)
        minibatch_kmeans.fit(self.data)
        #print minibatch_kmeans.labels_
        #print self.labels

        return self.report(self.labels, minibatch_kmeans.labels_), minibatch_kmeans.labels_

    def report(self, labels, pred):
        homogeneity = homogeneity_score(labels, pred)
        completeness = completeness_score(labels, pred)
        v_measure = v_measure_score(labels, pred)
        fowlkes_mallows = fowlkes_mallows_score(labels, pred)
        adjusted_rand = adjusted_rand_score(labels, pred)
        adjusted_mutual = adjusted_mutual_info_score(labels, pred)

        #rpt = '%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'.format(homogeneity, completeness, v_measure, fowlkes_mallows, adjusted_rand, adjusted_mutual)
        
        rpt = [homogeneity, completeness, v_measure, fowlkes_mallows, adjusted_rand, adjusted_mutual]
        return rpt