# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:12:05 2021

@author: Gobinda
"""

import os

os.environ['KMP_DUPLICATE_LTB_OK'] = 'True'

basePath = \
    {
        'train':'C:/Users/Gobinda/Desktop/QDRML/data/GoogleDataImages_test',
        'test':'C:/Users/Gobinda/Desktop/QDRML/data/GoogleDataImages_train'
    }

lr = 1e-4


savePath = 'savedModel'

       
os.makedirs(savePath, exist_ok = True)

imageSize = (64, 64)


seed = 0

numEpochs = 5

testEveryEpochs = 1

batchSize = {
    'train':40,
    'test':40
    
    }
numWorkers = {
    
    'train':6,
    'test':6
    
    }
iterations = {
    
    'train':1000,
    'test':100
    
    }     