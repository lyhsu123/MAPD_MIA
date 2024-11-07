import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import math
from torchvision import models
from model_data_base import Net_cifar100_m, Net_mnist, Net_cifar10, dataloader
from utils import X_Y_depart, data_tgt_test_class, distance_direction
from FL_base_function import FL_Train
from robustness_estimate import hsja, tgtlist
from MAP_attack import projected_adv_attack, pro_dis_dir
class Arguments:
    def __init__(self, dataset_name):
        self.N_client_max = 5
        self.N_client = 5
        self.data_name = 'cifar100' if dataset_name == 'cifar100' else 'MNIST' if dataset_name == 'mnist' else 'cifar10'
        self.global_epoch = 10 if dataset_name == 'cifar100' else 10 if dataset_name == 'mnist' else 10
        self.local_epoch = 10 if dataset_name == 'cifar100' else 5 if dataset_name == 'mnist' else 5
        self.save_all_models = False
        self.local_lr = 0.001 if dataset_name == 'cifar100' else 0.01 if dataset_name == 'mnist' else 0.001
        self.train_with_test = True
        self.use_gpu = True
        self.target_client = np.random.randint(0, self.N_client)
        self.cuda_state = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_cpu = torch.device('cpu')
        self.image_shape = (3,32,32) if dataset_name == 'cifar100' else (1,28,28) if dataset_name == 'mnist' else (3,32,32)
        self.class_num = 100 if dataset_name == 'cifar100' else 10 if dataset_name == 'mnist' else 10
        self.adv_iter = 60
        self.adv_lr = 0.3 if dataset_name == 'cifar100' else 0.3 if dataset_name == 'mnist' else 0.3
        self.decay_lr = 1
        self.num_evals_boundary = 5000
        self.beta_init = 0
        self.beta_step_size = 0.1
        self.beta_max = 2
        self.adv_with_test = True
        self.sample_num = 1000
        self.radius_num = 5
        self.noise_radnum = 100
        self.clear_num = 1036 if dataset_name == 'cifar100' else 2096 if dataset_name == 'mnist' else 1411
         # for tear
        self.target_num = 10

dataset_name = 'cifar10'
client_loaders, train_loader, test_loader = dataloader(dataset_name)
X_trainset, Y_trainset = X_Y_depart(train_loader, mode='train')    #torch.Size([50000, 3, 32, 32])    torch.Size([50000])    
X_testset, Y_testset = X_Y_depart(test_loader, mode='test')   #torch.Size([10000, 3, 32, 32])     torch.Size([10000])
if dataset_name == 'cifar100':
    FL_params = Arguments(dataset_name)
#     resnet = models.resnet18(pretrained=True)
#     resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(2,2), padding=(3, 3), bias=False)
#     init_GM = resnet  #training acc: 0.73     test acc: 0.43    drawback: hsja is very slow
#     init_GM = Net_cifar100_s()   #training acc: 0.3801           test acc: 0.2089    drawback: tgt_list doesn't have all the classes
    init_GM = Net_cifar100_m()
elif dataset_name == 'mnist':
    FL_params = Arguments(dataset_name)
    init_GM = Net_mnist()
elif dataset_name == 'cifar10':
    FL_params = Arguments(dataset_name)
    init_GM = Net_cifar10()

# init_GM.load_state_dict(torch.load('./data/cifar100/model/model_10.pth'))
model_path = './data/' + str(FL_params.data_name) + '/model/'
all_GMs, all_LMs = FL_Train(init_GM, client_loaders, test_loader, FL_params)
torch.save(all_GMs[-1].state_dict(), model_path+"model_"+str(FL_params.global_epoch)+".pth")

tgt_loader = client_loaders[FL_params.target_client]
init_GM.load_state_dict(torch.load('./data/' + str(FL_params.data_name) + '/model/model_'+str(FL_params.global_epoch)+'.pth'))
model = copy.deepcopy(init_GM.cuda())
FL_params.clear_num = data_tgt_test_class(tgt_loader, test_loader, model, FL_params)

clear_num = FL_params.clear_num
model.eval()
X_tgt_dataset = torch.load('./data/' + str(FL_params.data_name) + '/X_tgt_class_{}.pth'.format(clear_num))    #(samnum, *shape)
Y_tgt_dataset = torch.load('./data/' + str(FL_params.data_name) + '/Y_tgt_class_{}.pth'.format(clear_num))    #(samnum,)
X_test_dataset = torch.load('./data/' + str(FL_params.data_name) + '/X_test_class_{}.pth'.format(clear_num))  #(samnum, *shape)
Y_test_dataset = torch.load('./data/' + str(FL_params.data_name) + '/Y_test_class_{}.pth'.format(clear_num))  #(samnum,)
X_trainset = torch.load('./data/' + str(FL_params.data_name) + '/X_trainset.pth')              #(allnum, *shape)
Y_trainset = torch.load('./data/' + str(FL_params.data_name) + '/Y_trainset.pth')              #(allnum,)
tgt_list = tgtlist(model, X_trainset, Y_trainset, FL_params)
num = FL_params.sample_num    # 400
num_iter = 50   # 80
Adv_sample_train = hsja(model, X_tgt_dataset[:num], num=num, num_iter=num_iter, Y_dataset=Y_tgt_dataset[:num], tgt_list=tgt_list, mode='train', FL_params=FL_params)
Adv_sample_test = hsja(model, X_test_dataset[:num], num=num, num_iter=num_iter, Y_dataset=Y_test_dataset[:num], tgt_list=tgt_list, mode='test', FL_params=FL_params)

adv_samples_train = torch.load('./data/' + str(FL_params.data_name) + '/hsja_adv_sample_train+1000.pth')
adv_samples_test = torch.load('./data/' + str(FL_params.data_name) + '/hsja_adv_sample_test+1000.pth')

distances_train, adv_samples_train = distance_direction(adv_samples_train, X_tgt_dataset[:FL_params.sample_num], 0)
distances_test, adv_samples_test = distance_direction(adv_samples_test, X_test_dataset[:FL_params.sample_num], 0)
adv_min_train = adv_samples_train[torch.argmin(distances_train, dim=-1), torch.arange(adv_samples_train.shape[1])]  #(samnum, *shape)
adv_min_test = adv_samples_test[torch.argmin(distances_test, dim=-1), torch.arange(adv_samples_test.shape[1])]    #(samnum, *shape)
torch.save(distances_train, './data/' + str(FL_params.data_name) + '/' + str(FL_params.data_name) + '_distrain.pth')
torch.save(distances_test, './data/' + str(FL_params.data_name) + '/' + str(FL_params.data_name) + '_distest.pth')
torch.save(adv_min_train, './data/' + str(FL_params.data_name) + '/' + str(FL_params.data_name) + '_advmintrain.pth')
torch.save(adv_min_test, './data/' + str(FL_params.data_name) + '/' + str(FL_params.data_name) + '_advmintest.pth')

sample_iter = 400
radius_num = 5
noise_radnum = 100
projected_advs_mem = torch.zeros((FL_params.sample_num, radius_num*noise_radnum, *FL_params.image_shape))
projected_advs_nonmem = torch.zeros((FL_params.sample_num, radius_num*noise_radnum, *FL_params.image_shape))
for idx in range(radius_num):
    for idt in range(math.ceil(FL_params.sample_num / sample_iter)):
        print('radius_num: {}/{}, process iter: {}/{}'.format(idx+1, radius_num, idt+1, math.ceil(FL_params.sample_num / sample_iter)))
        if idt == math.ceil(FL_params.sample_num / sample_iter)-1:
            projected_advs_mem[idt*sample_iter:, idx*noise_radnum:(idx+1)*noise_radnum] = projected_adv_attack(model, adv_min_train[idt*sample_iter:FL_params.sample_num], X_tgt_dataset[idt*sample_iter:FL_params.sample_num], FL_params, noise_num=noise_radnum, radius=1.5*(idx+1), random_seed=315, tol=0.000015)
            projected_advs_nonmem[idt*sample_iter:, idx*noise_radnum:(idx+1)*noise_radnum] = projected_adv_attack(model, adv_min_test[idt*sample_iter:FL_params.sample_num], X_test_dataset[idt*sample_iter:FL_params.sample_num], FL_params, noise_num=noise_radnum, radius=1.5*(idx+1), random_seed=315, tol=0.000015)
        else:
            projected_advs_mem[idt*sample_iter:(idt+1)*sample_iter, idx*noise_radnum:(idx+1)*noise_radnum] = projected_adv_attack(model, adv_min_train[idt*sample_iter:(idt+1)*sample_iter], X_tgt_dataset[idt*sample_iter:(idt+1)*sample_iter], FL_params, noise_num=noise_radnum, radius=1.5*(idx+1), random_seed=315, tol=0.000015)
            projected_advs_nonmem[idt*sample_iter:(idt+1)*sample_iter, idx*noise_radnum:(idx+1)*noise_radnum] = projected_adv_attack(model, adv_min_test[idt*sample_iter:(idt+1)*sample_iter], X_test_dataset[idt*sample_iter:(idt+1)*sample_iter], FL_params, noise_num=noise_radnum, radius=1.5*(idx+1), random_seed=315, tol=0.000015)
torch.save(projected_advs_mem, './data/' + str(FL_params.data_name) + '/projected_advs_mem.pth')
torch.save(projected_advs_nonmem, './data/' + str(FL_params.data_name) + '/projected_advs_nonmem.pth')


projected_advs_mem = torch.load('./data/' + str(FL_params.data_name) + '/projected_advs_mem.pth')
projected_advs_nonmem = torch.load('./data/' + str(FL_params.data_name) + '/projected_advs_nonmem.pth')
print('load dataset finish!')
pro_dis_dir(projected_advs_mem, X_tgt_dataset[:FL_params.sample_num], mode='mem', FL_params=FL_params)
pro_dis_dir(projected_advs_nonmem, X_test_dataset[:FL_params.sample_num], mode='nonmem', FL_params=FL_params)
