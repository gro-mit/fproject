#coding=utf-8

import numpy as np
#import rpy2.robjects as robjects
from random import randint
import time
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append('..')
from fe_exp.utils import *

def k_select(data, k_max):
    sse = []
    sse_slope = [0, 0]
    silhouette = [0, 0]

    for k in range(2, k_max+1):
        kmeans = cluster.KMeans(n_clusters = k, init = 'k-means++')
        kmeans.fit(data)
        pred = kmeans.labels_
        sse.append(kmeans.inertia_)
        s_score = silhouette_score(data, pred, metric='euclidean')
        silhouette.append(s_score)

    for i in range(len(sse) - 1):
        slope = sse[i+1] - sse[i]
        sse_slope.append(slope)

    '''
    robjects.r.source('./kcomp_functions.r')
    pb_res = robjects.r.find_clusters_scan(data, k_max)
    inffered_k = pb_res.rx('nclusters')
    '''
    return sse, sse_slope, silhouette

def work_flow(datapath, labelspath):
    data, labels = load_data(datapath, labelspath)
    prepare = PreProcessing(data)
    data = prepare.standarization()
    class_num = len(set(labels))
    print
    print 'raw    dimension: {0}'.format(data.shape)
    print 'true class number: {0}'.format(class_num)

    sse, sse_slope, s_score = k_select(data, 9)
    print '-----------------------------------'
    print sse
    print s_score
    print '-----------------------------------'
    print 
    print sse_slope.index(min(sse_slope))
    print s_score.index(max(s_score))
    print
        
    
if __name__ == '__main__':
    dataset = ['ecoli', 'glass', 'iris', 'wine', 'ionosphere']
    res_name = './result/' + time.strftime('%m%d%H%M', time.localtime()) + 'kselect'
    dataset_path = './data/'
    for ds_name in dataset:
        print ds_name
        datapath = dataset_path + ds_name + '.data'
        labelspath = dataset_path + ds_name + '.labels'
        work_flow(datapath, labelspath)



