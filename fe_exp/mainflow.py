#coding=utf-8

import numpy as np
from random import randint
import time
from utils import *

def work_flow(res_name, data, labels, ratio):
    with open(res_name, 'w') as f:
        f.write('fs\taccuracy\truntime\tprecision-\1recall-\1f1score-\1support-\1precision+\1recall+\1f1score+\1support+\n')

    print
    prepare = PreProcessing(data)
    data = prepare.standarization()
    print 'raw    dimension: {0}'.format(data.shape)
    print 'target dimension: ({0}, {1})'.format(data.shape[0], int(data.shape[1] * ratio))
    
#    fs_method = ['base', 'l1', 'l2', 'tree', 'MI', 'ftest']
    fs_method = ['base', 'deep']
#    fs_method = ['deep']
    clf_method = 'knn'
    repeat_times = 10 

    for fs_name in fs_method:
        print '---------------------------------------------------{0}'.format(fs_name)
        acc_box = []
        report_box0 = []
        report_box1 = []
        
        runtime_box = []
        for i in xrange(repeat_times):
            start_time = time.clock()
            fs = FeatureSelection(method = fs_name, ratio = ratio, random_state = randint(0,99999), data = data, labels = labels)
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

        single_res = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\n'.format(fs_name, mean_acc, mean_runtime, mean_report[0][0], mean_report[0][1], mean_report[0][2], mean_report[0][3], mean_report[1][0], mean_report[1][1], mean_report[1][2], mean_report[1][3])
        with open(res_name, 'a+') as f:
            f.write(single_res)

if __name__ == '__main__':
    dataset_name = 'arcene'
    ratio = float(sys.argv[1])
    print dataset_name
    print ratio
    res_name = './result/' + time.strftime('%m%d%H%M', time.localtime()) + dataset_name + sys.argv[1]
    dataset_path = './data/' + dataset_name
    datapath = dataset_path + '.data'
    labelspath = dataset_path + '.labels'

    data, labels = load_data(datapath, labelspath)
    work_flow(res_name, data, labels, ratio)
