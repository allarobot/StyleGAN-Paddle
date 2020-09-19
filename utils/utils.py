#!/usr/bin/env python
# coding: utf-8
"""
    @author: samuel ko
"""
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2

import paddle.fluid as fluid
import logging


def LOG(filename="log.txt"):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger.info



def plotLossCurve(opts, Loss_D_list, Loss_G_list):
    plt.figure()
    plt.plot(Loss_D_list, '-')
    plt.title("Loss curve (Discriminator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_discriminator.png'))

    plt.figure()
    plt.plot(Loss_G_list, '-o')
    plt.title("Loss curve (Generator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_generator.png'))


def show_image(image,title):
    image = np.array(image).transpose((1,2,0))
    plt.imshow(image)
    plt.title("cat {}".format(title))

def save_image(image,path):
    image = np.array(image).transpose((1,2,0))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)*255.0
    LOG()(f"image {image.shape} saving to {path}")
    image = np.uint8(image)
    cv2.imwrite(path,image)

def print_model(x,m):
    for item in m.sublayers():
    # item是LeNet类中的一个子层
    # 查看经过子层之后的输出数据形状
        try:
            x = item(x)
        except:
            x = fluid.layers.reshape(x, [x.shape[0], -1])
            x = item(x)
        if len(item.parameters())==2:
            # 查看卷积和全连接层的数据和参数的形状，
            # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
            print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
        else:
            # 池化层没有参数
            print(item.full_name(), x.shape)

# def save_model(states:dict, path:str):
#     fp = open(path,'w')
#     file_paths = []
#     for model_name in states:
#         state_dict = states[model_name]
#         path,ext = os.path.splitext(path)
#         if ext:
#             file_path = path+"-"+model_name+"."+ext
#         else:
#             file_path = path+"-"+model_name
#         file_paths.append(file_path)
#         fluid.dygraph.save_dygraph(state_dict,file_path)
#     fp.writelines(file_paths)
#     fp.close()
    
# def load_model(path:str):
#     fp = open(path,'r')
#     file_paths = fp.readlines()
#     states={}
#     for path in file_paths:
#         model_name = os.path.splitext(path)[0].split('-')[-1]
#         states[model_name] = fluid.dygraph.load_model(path)

#     return states
import pickle
def save_checkpoint(states:dict, path:str):
    dump_states = {}
    for model_name in states:
        state_dict = states[model_name]
        if type(state_dict).__name__ == 'OrderedDict':
            pathname,ext = os.path.splitext(path)
            state_dict_path = pathname+"-"+model_name
            fluid.dygraph.save_dygraph(state_dict,state_dict_path)
            dump_states[model_name] = ('OrderedDict',state_dict_path)
        else:
            dump_states[model_name] = states[model_name]
    with open (path, 'wb') as f: 
        pickle.dump(dump_states, f)
    
def load_checkpoint(path:str):
    load_states = {}
    with open(path,'rb') as f:
        load_states= pickle.load(f)
    for k,v in load_states.items():
        if type(v) is tuple and v[0] == 'OrderedDict':
            state = fluid.dygraph.load_dygraph(v[1]) # state[0]: parameter_dict, state[1]:optimizer_dict
            load_states[k] = state[0]
    return load_states
