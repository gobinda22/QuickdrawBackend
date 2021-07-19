# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:28:31 2021

@author: Gobinda
"""

from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config




class dataloader(Dataset):
    def __init__(self, type_):
        
        self.type_ = type_
        self.allClasses = ['Bird', 'Flower', 'Hand', 'House', 'Pencil', 'Mug', 'Spoon', 'Sun', 'Tree', 'Umbrella']
        
        self.imagePaths = []
        self.targets = []
        
        for no,classI in enumerate(self.allClasses):
            imagePath = sorted(os.listdir(config.basePath[self.type_] + '/' + classI + '/image/')) #sorted is used for same output for different OS
            
            if self.type_ == 'train':
                imagePath = imagePath[0:int(0.9 * len(imagePath))] # 0 to 90% of the images
            else:
                imagePath = imagePath[int(0.9 * len(imagePath)):] #90 to 100% of the images
                
            for image in imagePath:#storing the all imagePaths and the targets
                self.imagePaths.append(config.basePath[self.type_] + '/' + classI + '/image/' + image)
                self.targets.append(no)
                
    def crop_image(self, image):

        y, x =  np.where(image!=0)#gives y and x coordinate when the image is nor zero
        y_min, x_min, y_max, x_max = np.min(y), np.min(x), np.max(y), np.max(x)
        image = image[y_min:y_max, x_min:x_max]
        return image


    def aspect_resize(self, image):
        max_shape = max(image.shape[0], image.shape[1])#image.shape[0] gives the height and 1 gives the width
        new_image = np.zeros([max_shape, max_shape])#take the maximum of width or height, if max is 60 then (60,60)
        y_min = (max_shape - image.shape[0])//2 #(60-40)/2
        y_max = y_min + image.shape[0] #(10+40)
        x_min = (max_shape - image.shape[1])//2
        x_max = x_min + image.shape[1]
        new_image[y_min:y_max, x_min:x_max] = image
        new_image = Image.fromarray(new_image)
        new_image = new_image.resize(config.imageSize)
        new_image = (np.array(new_image) > 0.1).astype(np.float32)
        return new_image
        
    def process(self, path):
        
        image = plt.imread(path)[:, :, 3]#coverting into grayscale image
        image = Image.fromarray(image)
        image = image.resize(config.imageSize)# convet it into (64,64)
        finalImage = (np.array(image)>0.1).astype(np.float32)#to make float value to 0 and 1
        finalImage = self.crop_image(finalImage)
        finalImage = self.aspect_resize(finalImage)

        return finalImage        
        
    def __getitem__(self, item):
       image = self.process(self.imagePaths[item])
       #[batch_size, 1 , height , width] CHW model channel height and width

       return image[None, : , :], self.targets[item], self.imagePaths[item]

    def __len__(self):

        return len(self.imagePaths)   

                   
def getDataLoader(type_='train'):

        return DataLoader(
            dataloader(type_=type_),
            batch_size=config.batchSize[type_],
            num_workers=config.numWorkers[type_],
            shuffle=type_ == 'train'
         )