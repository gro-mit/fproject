#coding=utf-8

import numpy as np
import time
from utils import *

def work_flow(data, labels, ratio):
    print
    prepare = PreProcessing(data)
    data = prepare.standarization()
    print 'raw    dimension: {0}'.format(data.shape)
    print 'target dimension: ({0}, {1})'.format(data.shape[0], int(data.shape[1] * ratio))
    
    fs_method = ['base', 'l1', 'l2', 'tree', 'MI']
#    fs_method = ['base', 'tree']
    clf_method = 'knn'
    for fs_name in fs_method:
        print '---------------------------------------------------{0}'.format(fs_name)
        start_time = time.clock()
        fs = FeatureSelection(method = fs_name, ratio = ratio, data = data, labels = labels)
        kf_data, test_data, kf_labels, test_labels = fs.excute()
        print 'fs part: {0} {1} sec'.format(fs_name, time.clock() - start_time)
        start_time = time.clock()
        clf = Classifier(method = clf_method, kf_data = kf_data, test_data = test_data, kf_labels = kf_labels, test_labels = test_labels)
        clf.excute()
        print 'clf part: {0} {1} sec'.format(clf_method, time.clock() - start_time)

if __name__ == '__main__':
    dataset_path = './data/arcene'
    datapath = dataset_path + '.data'
    labelspath = dataset_path + '.labels'
    ratio = 0.05

    data, labels = load_data(datapath, labelspath)
    work_flow(data, labels, ratio)
