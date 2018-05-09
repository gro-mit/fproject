#coding=utf-8

import numpy as np
from random import randint
import time
from utils import *


def pipe_line(data, labels, fs_method, fs_ratio, clf_method):
    ratios = [0.25, 0.5, 0.75]
    dr_method = ['base', 'pca', 'rp', 'mds']
    start_time = time.clock()
    fs = FeatureSelection(method = fs_method, ratio = fs_ratio, random_state = randint(0, 99999), data = data, labels = labels)
    kf_data, test_data, kf_labels, test_labels = fs.excute()
    fs_time = time.clock() - start_time

    res_set = []
    for dr_name in dr_method:
        dr_res = []
        #res format: [fs_time, dr_time0.25, total_time, acc0.25, report0.25[0], report0.25[1]] 12 items
        for dr_ratio in ratios:
            dr_time = time.clock()
            dr = Dimensionality_reduction(dr_name, dr_ratio, kf_data, test_data, kf_labels, test_labels)
            dr_kf_data, dr_test_data, dr_kf_labels, dr_test_labels = dr.excute()
            dr_time = time.clock() - dr_time
            one_dr_res = [fs_time, dr_time, fs_time + dr_time]
            clf = Classifier(method = clf_method, kf_data = dr_kf_data, test_data = dr_test_data, kf_labels = dr_kf_labels, test_labels = dr_test_labels)
            acc, report = clf.excute()
            one_dr_res.append(acc)
            one_dr_res.extend(report[0])
            one_dr_res.extend(report[1])
            dr_res.extend(one_dr_res)
        res_set.extend(dr_res)
    print len(res_set)
    return res_set



def work_flow(res_name, data, labels, ratio):
    #with open(res_name, 'w') as f:
    #    f.write('fs\taccuracy\truntime\tprecision-\1recall-\1f1score-\1support-\1precision+\1recall+\1f1score+\1support+\n')

    print
    prepare = PreProcessing(data)
    data = prepare.standarization()
    print 'raw    dimension: {0}'.format(data.shape)
    print 'target dimension: ({0}, {1})'.format(data.shape[0], int(data.shape[1] * ratio))
    
    fs_method = ['base', 'l1', 'l2', 'tree', 'MI', 'ftest']
    #fs_method = ['base', 'deep']
#    fs_method = ['deep']
    clf_method = 'knn'
    repeat_times = 10 

    for fs_name in fs_method:
        print '---------------------------------------------------{0}'.format(fs_name)
        
        res_box = []
        for i in xrange(repeat_times):
            '''
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
            '''
            res = pipe_line(data, labels, fs_name, ratio, clf_method)
            res_box.append(res)
        #show experiment mean result
        mean_res = np.mean(np.array(res_box), axis = 0)#dr 36
        tmp_res = []
        print '*************** mean report *************** {0}'.format(fs_name)
        s1 = '{0}\tbase\t0.25\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[0:12]))
        s2 = '{0}\tbase\t0.50\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[12:24]))
        s3 = '{0}\tbase\t0.75\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[24:36]))
        s4 = '{0}\tpca\t0.25\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[36:48]))
        s5 = '{0}\tpca\t0.50\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[48:60]))
        s6 = '{0}\tpca\t0.75\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[60:72]))
        s7 = '{0}\trp\t0.25\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[72:84]))
        s8 = '{0}\trp\t0.50\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[84:96]))
        s9 = '{0}\trp\t0.75\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[96:108]))
        s10 = '{0}\tmds\t0.25\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[108:120]))
        s11 = '{0}\tmds\t0.50\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[120:132]))
        s12 = '{0}\tmds\t0.75\t{1}\n'.format(fs_name, '\t'.join(str(item) for item in mean_res[132:144]))
        tmp_res.append(s1)
        tmp_res.append(s2)
        tmp_res.append(s3)
        tmp_res.append(s4)
        tmp_res.append(s5)
        tmp_res.append(s6)
        tmp_res.append(s7)
        tmp_res.append(s8)
        tmp_res.append(s9)
        tmp_res.append(s10)
        tmp_res.append(s11)
        tmp_res.append(s12)
        print s1,
        print s2,
        print s3,
        print s4,
        print s5,
        print s6,
        print s7,
        print s8,
        print s9,
        print s10,
        print s11,
        print s12,
        print '*******************************************'
        print 

        for info in tmp_res:
            with open(res_name, 'a+') as f:
                f.write(info)
        '''
        tmp_res = []
        dr_method = ['base', 'pca', 'rp', 'mds']
        dr_ratio = [0.25, 0.5, 0.75]
        for i in range(len(dr_method)):
            for j in range(len(dr_ratio)):
                tmp_res.extend(mean_res[(i*4+j)*12:(i*4+j+1)*12])
                info = '\t'.join(str(item) for item in tmp_res)
                single_res = '{0}\t{1}\t{2}\t{3}\n'.format(fs_name, dr_method[i], dr_ratio[j], info)
                with open(res_name, 'a+') as f:
                    f.write(single_res)
                tmp_res = []
        '''
        
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
