#coding=utf-8

import numpy as np
from utils import *

def work_flow(data, labels):
    prepare = PreProcessing(data)
    data = prepare.standarization()
    

    
    


if __name__ == '__main__':
    dataset_path = './data/arcene'
    datapath = dataset_path + '.data'
    labelspath = dataset_path + '.labels'

    data, labels = load_data(datapath, labelpath)
    work_flow(data, labels)
