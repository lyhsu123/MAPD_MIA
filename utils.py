# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:52:03 2021
"""

import torch
import numpy as np
from einops import rearrange

def X_Y_depart(dataloader, FL_params, mode='train'):
    X_dataset = list()
    Y_dataset = list()
    for img, label in dataloader:
        X_dataset.append(img)
        Y_dataset.append(label)
    X_dataset = torch.vstack(X_dataset)
    Y_dataset = torch.hstack(Y_dataset)
    torch.save(X_dataset, './data/' + str(FL_params.data_name) + '/X_{}set.pth'.format(mode))
    torch.save(Y_dataset, './data/' + str(FL_params.data_name) + '/Y_{}set.pth'.format(mode))
    return X_dataset, Y_dataset

def clear_image(data_loader, model, clabel):
    X_class = list()
    Y_class = list()
    for image, label in data_loader:
        for idl in range(len(label)):
            if label[idl] == torch.tensor(clabel) and torch.argmax(model(image[idl].unsqueeze(0).cuda())).cpu() == torch.tensor(clabel):
                X_class.append(image[idl].unsqueeze(0))
                Y_class.append(label[idl])
    if len(X_class)!=0:
        X_class = torch.cat(X_class, dim=0)
        Y_class = torch.tensor(Y_class)
    return X_class, Y_class, len(Y_class)

def data_tgt_test_class(tgt_loader, test_loader, model, FL_params):
    clear_num = 0
    X_tgtclass = list()
    X_testclass = list()
    Y_tgtclass = list()
    Y_testclass = list()
    model.eval()   #不加这句话，resnet的dropout仍会运作，会导致模型基本没有预测率
    for idx in range(FL_params.class_num):
        X_tgt_dataset, Y_tgt_dataset, clear_tgtnum = clear_image(tgt_loader, model, idx)
        X_test_dataset, Y_test_dataset, clear_testnum = clear_image(test_loader, model, idx)
        samnum = torch.min(torch.tensor([clear_tgtnum, clear_testnum]))
        clear_num += samnum
        if samnum !=0:
            X_tgtclass.append(X_tgt_dataset[:samnum])
            X_testclass.append(X_test_dataset[:samnum])
            Y_tgtclass.append(Y_tgt_dataset[:samnum])
            Y_testclass.append(Y_test_dataset[:samnum])
        print('class {}: sample number {}'.format(idx, samnum))
        if clear_num > FL_params.sample_num:
            X_tgtclass = torch.cat(X_tgtclass, dim=0)
            X_testclass = torch.cat(X_testclass, dim=0)
            Y_tgtclass = torch.cat(Y_tgtclass, dim=0)
            Y_testclass = torch.cat(Y_testclass, dim=0)
            print(X_tgtclass.shape)
            print(Y_tgtclass.shape)
            print(X_testclass.shape)
            print(Y_testclass.shape)
            break
    torch.save(X_tgtclass, './data/' + str(FL_params.data_name) + '/X_tgt_class_{}.pth'.format(clear_num))
    torch.save(Y_tgtclass, './data/' + str(FL_params.data_name) + '/Y_tgt_class_{}.pth'.format(clear_num))
    torch.save(X_testclass, './data/' + str(FL_params.data_name) + '/X_test_class_{}.pth'.format(clear_num))
    torch.save(Y_testclass, './data/' + str(FL_params.data_name) + '/Y_test_class_{}.pth'.format(clear_num))
    return clear_num

def distance_direction(adv_sample, ori_sample, label):
    ori_sample = ori_sample.unsqueeze(0)   #(1, samnum, *shape)
    adv_sample = adv_sample[torch.arange(adv_sample.shape[0])!=label]   #(classnum-1, samnum, *shape)
    sub_sample = (adv_sample - ori_sample).view(adv_sample.shape[0], adv_sample.shape[1], -1)  #(classnum-1, samnum, -1)
    distances = torch.norm(sub_sample, p=2, dim=2)   #(classnum-1, samnum)
    directions = sub_sample / distances.unsqueeze(2)  #(classnum-1, samnum, -1)
    distances = rearrange(distances, 'i j -> j i')   #(samnum, classnum-1)
    return distances, adv_sample