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
    repeat_times = 5

    for fs_name in fs_method:
        print '---------------------------------------------------{0}'.format(fs_name)
        acc_box = []
        report_box0 = []
        report_box1 = []
        
        runtime_box = []
        for i in xrange(repeat_times):
            start_time = time.clock()
            fs = FeatureSelection(method = fs_name, ratio = ratio, data = data, labels = labels)
            kf_data, test_data, kf_labels, test_labels = fs.excute()
            fs_runtime = time.clock() - start_time
            runtime_box.append(fs_runtime)
            #print 'fs part: {0} {1} sec'.format(fs_name, fs_runtime)

            start_time = time.clock()
            clf = Classifier(method = clf_method, kf_data = kf_data, test_data = test_data, kf_labels = kf_labels, test_labels = test_labels)
            acc, report = clf.excute()
            acc_box.append(acc)
            report_box0.append(report[0])
            report_box1.append(report[1])
            #print 'clf part: {0} {1} sec'.format(clf_method, time.clock() - start_time)

        #show experiment mean result
        mean_acc = np.mean(acc_box)
        report_box0 = np.array(report_box0)
        report_box1 = np.array(report_box1)
        mean_report = np.vstack([np.mean(report_box0, axis = 0), np.mean(report_box1, axis = 0)])
        mean_runtime = np.mean(runtime_box)
        print '*************** mean report ***************'
        print 'runtime: {0} sec'.format(mean_runtime)
        print 'acc:     {0}'.format(mean_acc)
        print 'precision\trecall\tf1-score\tsupport'
        print mean_report[0]
        print mean_report[1]
        print '*******************************************'
        print 

if __name__ == '__main__':
    dataset_path = './data/arcene'
    datapath = dataset_path + '.data'
    labelspath = dataset_path + '.labels'
    ratio = 0.05

    data, labels = load_data(datapath, labelspath)
    work_flow(data, labels, ratio)
