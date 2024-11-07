import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import os
os.chdir(os.path.dirname(__file__))


# =============================================================================
# 1.server的全局训练迭代次数一共11次，每次全局训练完后，server将会收集所有（5个）client训练好的模型参数进行平均，
# 来更新全局模型
# 2.每个client的模型由server的全局模型分发得到，并采用同样的算法进行训练，每个client将使用本地数据集进行训练迭代10次
# 3.FL_Train这个函数将会返回两个模型集合：全局模型集合（一共12个，包括最开始初始化的全局模型和11次全局训练后得到的
# 全局模型）， 本地模型集合（包括了每次全局训练后得到的5个client各自的本地模型，所以一共有55个本地模型）（注：这个集合
# 在代码中从来没有用过）
# 4.每个client在本地训练完一个epoch后都会计算一次在本地训练集上的准确率 和 在测试集上的准确率
# 5.server在全局训练完一个epoch后都会计算一次在目标训练集（好奇的client，通过随机来选出）上的准确率 和 在测试集上的
# 准确率。注意所有的测试集都是同一个样本集，整个联邦学习共用一个测试集
# 6.全局模型最终都会保存
# =============================================================================
def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):

    print("FL Training Starting...")
    all_global_models = list()
    all_client_models = list()

    global_model = init_global_model

    all_global_models.append(copy.deepcopy(global_model))
    
    if FL_params.save_all_models:
        model_path = "./model/"+str(FL_params.data_name)+"/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
    for epoch in range(FL_params.global_epoch):
        
        # Save the global models
        if FL_params.save_all_models:
            model_ = copy.deepcopy(global_model)
            torch.save(model_.state_dict(), model_path+"model_"+str(epoch)+".pth")
        
        print("Global Federated Learning epoch = {}/{}".format(epoch + 1, FL_params.global_epoch))
        # Each client performs model training locally
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params, epoch+1)
        # The server performs fedavg and updates global model parameters
        global_model = fedavg(client_models)
        # Test the global models
        if FL_params.train_with_test:
            test(global_model, client_data_loaders[FL_params.target_client], test_loader, FL_params)

        all_global_models.append(copy.deepcopy(global_model))
        all_client_models += client_models
        
    if FL_params.save_all_models:
        model_ = copy.deepcopy(global_model)
        torch.save(model_.state_dict(), model_path+"model_"+str(FL_params.global_epoch)+".pth")
    
    print("FL Training Successfully!")
    print()

    return all_global_models, all_client_models


def global_train_once(global_model, client_data_loaders, test_loader, FL_params, global_epoch):

    # set the initial model and optimizer of each client
    client_models = []
    client_optims = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_optims.append(optim.Adam(client_models[ii].parameters(), lr=FL_params.local_lr))

    for client_idx in range(FL_params.N_client):
        model = client_models[client_idx]
        optimizer = client_optims[client_idx]

        model.to(FL_params.device)
        model.train()

        # local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data, target = data.to(FL_params.device), target.to(FL_params.device)

                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()

            
            print("Local Client No. {}/{}, Local Epoch: {}/{}, Global Epoch:{}/{}".format(client_idx+1, 
                    FL_params.N_client, local_epoch+1, FL_params.local_epoch, global_epoch, FL_params.global_epoch))
            if FL_params.train_with_test:
                test(model, client_data_loaders[client_idx], test_loader, FL_params)

        client_models[client_idx] = model

    return client_models


def test(model, train_loader, test_loader, FL_params):

    model = model.to(FL_params.device_cpu)  #test时用了cpu
    model.eval()

    train_loss = 0
    train_acc = 0
    for data, target in train_loader:
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        train_loss += criteria(output, target)  # sum up batch loss
        pred = torch.argmax(output, dim=1)
        train_acc += accuracy_score(pred, target)

    train_loss /= len(train_loader.dataset)
    train_acc = train_acc / np.ceil(len(train_loader.dataset) / train_loader.batch_size)
    # print('Train set: Average loss: {:.4f}'.format(train_loss))
    print('Train set: Average acc:  {:.4f}'.format(train_acc))

    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        test_loss += criteria(output, target)  # sum up batch loss

        pred = torch.argmax(output, dim=1)
        test_acc += accuracy_score(pred, target)

    test_loss /= len(test_loader.dataset)
    test_acc = test_acc / np.ceil(len(test_loader.dataset) / test_loader.batch_size)
    # print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Test set: Average acc:  {:.4f}'.format(test_acc))

    model = model.to(FL_params.device)
    model.train()


def fedavg(local_models):
    """
    :param local_models: list of local models
    :return: update_global_model: global model after fedavg
    """
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] = avg_state_dict[layer].float() / len(local_models)

    global_model.load_state_dict(avg_state_dict)
    return global_model
