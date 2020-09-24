#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import cv2
import numpy as np
from paddle.fluid.io import DataLoader
import os
from paddlex.cls import transforms
#import Albumentation as A

from glob import glob

import paddle.fluid as fluid

def loader(path):
    x  = Image.open(path).convert("RGB")
    x = np.asarray(x).astype('float32')
    #x = cv2.imread("work/"+path,cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)/255.0
    x = (cv2.resize(x,(1024,1024))-0.5)/0.5
    return x

def loader_test(path):
    x  = Image.open(path).convert("RGB")
    x = np.asarray(x).astype('float32')
    #x = cv2.imread("work/"+path,cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)/255.0
    x = (cv2.resize(x,(1024,1024))-0.5)/0.5
    return x

transform_ops = transforms.Compose(
    [
    #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])   # this will do 1/255.0
    transforms.RandomHorizontalFlip(prob=0.5),
    transforms.RandomRotate(rotate_range=30, prob=0.5),
    transforms.RandomCrop(crop_size=224, lower_scale=0.7, lower_ratio=3. / 4, upper_ratio=4. / 3),
    transforms.RandomDistort(brightness_range=0.1, brightness_prob=0.5, contrast_range=0.1, contrast_prob=0.5, saturation_range=0.1, saturation_prob=0.0, hue_range=0.1, hue_prob=0.0)
    ]
)

def transform(img):
    #print("before transform: ",img.shape)
    img = transform_ops(img)[0]
    #print("after transform: ",img.shape)
    return img

# class TrainDataset:
#     def __init__(self,data):
#         self._data = data
#         self.size = data.shape[0]
#         self.cur = 0

#     def __len__(self):
#         return self.size
    
#     def __getitem__(self,idx):
#         img, label = loader(self._data[idx,0]),self._data[idx,1]
#         img = transform(img)
#         img = img.transpose(2,0,1)
#         return img, label
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.cur<self.size:
#             idx = self.cur
#             self.cur = self.cur + 1
#             return self.__getitem__(idx)
#         raise StopIteration

class Dataset:
    def __init__(self,data,transforms=None):
        self._data = data
        self.size = len(data)
        self.cur = 0
        self._transform = transforms

    def __len__(self):
        return self.size
    
    def __getitem__(self,idx):
        img = loader_test(self._data[idx])
        if self._transform:
            img = self._transform(img)
        img = img.transpose(2,0,1)
        #print(img.size,img.shape)  #3145728 (3, 1024, 1024)
        return img
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur<self.size:
            idx = self.cur
            self.cur = self.cur + 1
            return self.__getitem__(idx)
        raise StopIteration


# ## batch/shuffle dataset with Reader
def data_loader(path):
    search_pattern = os.path.join(path,"*.png")
    test_list_temp = glob(search_pattern) 
    test_list = []
    for path in test_list_temp:
        test_list.append(path)

    ## create Dataset
    dataset = Dataset(test_list) 
    def reader():
        for i in range(len(dataset)):
            yield dataset[i]

    shuffled_reader = fluid.io.shuffle(reader,100)    # shuffle buffer = 100
    batch_reader = fluid.io.batch(shuffled_reader,1)  # batch size = 1
    return batch_reader
