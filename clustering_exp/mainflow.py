#coding=utf-8

import numpy as np
from random import randint
import time
import sys
sys.path.append('..')
from fe_exp.utils import *

def work_flow(res_name, data, labels, ratio):
    #with open(res_name, 'w') as f:
     #   f.write('cluster\taccuracy\truntime\tprecision-\1recall-\1f1score-\1support-\1precision+\1recall+\1f1score+\1support+\n')

    print 'tree based fs method, ratio is {0}%'.format(ratio*100.0)
    prepare = PreProcessing(data)
    data = prepare.standarization()
    print 'raw    dimension: {0}'.format(data.shape)
    print 'target dimension: ({0}, {1})'.format(data.shape[0], int(data.shape[1] * ratio))
    
    fs = FeatureSelection(method = 'tree', ratio = ratio, random_state = 0, data = data, labels = labels)
    kf_data, test_data, kf_labels, test_labels = fs.excute()
    data = np.vstack([kf_data, test_data])
    labels = np.hstack([kf_labels, test_labels])
    cluster_method = ['kmeans', 'kmeans++', 'minibatchkmeans']
    measure = ['pearson', 'euclidean', 'cosine']
    repeat_times = 10

    for measure_name in measure:
        print '---------------------------------------------------{0}'.format(measure_name)
        runtime_box = []
        hc_report = []
        for i in range(repeat_times):
            start_time = time.clock()
            hc = Clustering(method = 'hc', measure = measure_name, n_clusters = 2, data = data, labels = labels)
            res, _ = hc.excute()
            hc_runtime = time.clock() - start_time
            runtime_box.append(hc_runtime)
            hc_report.append(res)

        hc_report = np.array(hc_report)
        mean_report = np.mean(hc_report, axis = 0)
        mean_runtime = np.mean(runtime_box)

        print '*************** mean report ***************'
        print 'runtime: {0} sec'.format(mean_runtime)
        print 'homogeneity\tcompleteness\tv_measure\tfowlkes_mallows\tadjusted_rand\tadjusted_mutual'
        print mean_report
        print '*******************************************'
        print 

        single_res = 'hc\t{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(measure_name, mean_runtime, mean_report[0], mean_report[1], mean_report[2], mean_report[3], mean_report[4], mean_report[5])
        with open(res_name, 'a+') as f:
            f.write(single_res)


    kmpp_pred = []
    mbkm_pred = []
    for cluster_name in cluster_method:
        print '---------------------------------------------------{0}'.format(cluster_name)
        runtime_box = []
        clu_report = []

        for i in range(repeat_times):
            start_time = time.clock()
            clu = Clustering(method = cluster_name, measure = None, n_clusters = 2, data = data, labels = labels)
            res, tmp_pred = clu.excute()
            clu_runtime = time.clock() - start_time
            runtime_box.append(clu_runtime)
            clu_report.append(res)
            if cluster_name == 'kmeans++':
                kmpp_pred.append(tmp_pred)
            elif cluster_name == 'minibatchkmeans':
                mbkm_pred.append(tmp_pred)

        clu_report = np.array(clu_report)
        mean_report = np.mean(clu_report, axis = 0)
        mean_runtime = np.mean(runtime_box)

        print '*************** mean report ***************'
        print 'runtime: {0} sec'.format(mean_runtime)
        print 'homogeneity\tcompleteness\tv_measure\tfowlkes_mallows\tadjusted_rand\tadjusted_mutual'
        print mean_report
        print '*******************************************'
        print 

        single_res = 'km\t{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(cluster_name, mean_runtime, mean_report[0], mean_report[1], mean_report[2], mean_report[3], mean_report[4], mean_report[5])
        with open(res_name, 'a+') as f:
            f.write(single_res)

    kmpp_pred = np.array(kmpp_pred)
    mbkm_pred = np.array(mbkm_pred)
    difference = kmpp_pred + mbkm_pred
    diff_index = [min(np.argwhere(difference[i] == 1).size, int(data.shape[0]-np.argwhere(difference[i] == 1).size))  for i in range(repeat_times)]
    print diff_index
    with open(res_name, 'a+') as f:
        f.write('\n')
        f.write('{0}\n'.format(diff_index))

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    ratio = float(sys.argv[2])
    print dataset_name
    print ratio
    res_name = './result/' + time.strftime('%m%d%H%M', time.localtime()) + dataset_name + sys.argv[2]
    dataset_path = './data/' + dataset_name
    datapath = dataset_path + '.data'
    labelspath = dataset_path + '.labels'

    data, labels = load_data(datapath, labelspath)
    work_flow(res_name, data, labels, ratio)
