#coding=utf-8

import numpy as np
from utils import *

def work_flow(data, labels):
    ratio = 0.1
    prepare = PreProcessing(data)
    data = prepare.standarization()
    
#    fs_method = ['base', 'l1', 'l2', 'tree', 'MI']
    fs_method = ['tree']
    for fs_name in fs_method:
        print 'fs method: ' + fs_name
        fs = FeatureSelection(method = fs_name, ratio = ratio, data = data, labels = labels)
        kf_data, test_data, kf_labels, test_labels = fs.excute()
        clf = Classifier(method = 'knn', kf_data = kf_data, test_data = test_data, kf_labels = kf_labels, test_labels = test_labels)
        clf.excute() 

if __name__ == '__main__':
    dataset_path = './data/arcene'
    datapath = dataset_path + '.data'
    labelspath = dataset_path + '.labels'

    data, labels = load_data(datapath, labelspath)
    work_flow(data, labels)
