import numpy as np
import torch
import math
import foolbox as fb
import foolbox.criteria as criteria
import math
import copy

def tgtlist(model, X_trainset, Y_trainset, FL_params):
    tgt_list = list()
    for tgt_label in range(FL_params.class_num):
        tgt_label = torch.tensor(tgt_label)
        for idb in range(len(X_trainset)):
            if Y_trainset[idb] == tgt_label and torch.argmax(model(X_trainset[idb].unsqueeze(0).cuda())).cpu() == tgt_label:
                tgt_tgt = X_trainset[idb]
                tgt_list.append(tgt_tgt)
                break
            if idb == len(X_trainset)-1:
                print('tgt_label:', tgt_label)
    assert len(tgt_list) == FL_params.class_num
    tgt_list = torch.stack(tgt_list,dim=0)
    print(tgt_list.shape)    #(classnum, *shape)
    return tgt_list
def is_adversarial(model, perturbed, adv_label, FL_params):
    predict_label = torch.argmax(model(perturbed.cuda()), dim=-1).cpu()
    return (predict_label == adv_label)   #(tarnum, )
def binary_search0(model, x_0, x_random, ori_label, adv_label, FL_params, tol=0.00001):
    adv = copy.deepcopy(x_random)   #(tarnum, *shape)
    cln = copy.deepcopy(x_0)    #(tarnum, *shape)
    old_mid = adv
    while True:
        mid = (cln + adv) / 2.0    #(tarnum, *shape)
        err = torch.argmax(torch.norm((adv - cln).reshape((adv.shape[0],-1)), p=2, dim=-1))
#         print(torch.norm(adv[err]-cln[err], p=2))
        if torch.all(torch.norm((adv - cln).reshape((adv.shape[0],-1)), p=2, dim=-1) < tol) or torch.all(mid == old_mid):
            is_adv = is_adversarial(model, adv, adv_label, FL_params)
            if not torch.any(is_adv):
                print('not')
                not_adv = torch.nonzero(is_adv==0, as_tuple=True)[0]
                adv[not_adv] = binary_search0(model, adv[not_adv], x_random[not_adv], torch.argmax(model(adv[not_adv].cuda())).cpu(), adv_label[not_adv], FL_params)
            break
        is_ori = is_adversarial(model, mid, ori_label, FL_params)
        adv_ind = torch.where(is_ori, 1, 0).reshape((is_ori.shape[0], *[1]*len(FL_params.image_shape)))
        adv = adv_ind * adv + (1 - adv_ind) * mid
        cln = adv_ind * mid + (1 - adv_ind) * cln
        old_mid = mid
    return adv
def decision_function(model, images, adv_labels, FL_params):
    predict_label = torch.argmax(model(images.cuda()), dim=1).unsqueeze(1).cpu()    #(tarnum * noinum, 1)
    target_label = adv_labels.repeat_interleave(int(len(images) / len(adv_labels))).unsqueeze(1)    #(tarnum * noinum, 1)
    return (predict_label == target_label).reshape(len(adv_labels), -1)      #(tarnum, noinum)
def gradient_estimation_boundary(model, samples, adv_labels, distances, FL_params):
    num_evals = FL_params.num_evals_boundary
    deltas = 1 / (len(samples.shape)-1) * distances    #(tarnum,)
    noise_shape = [samples.shape[0], num_evals, *FL_params.image_shape]
    rvs = torch.randn(*noise_shape)   #(tarnum, noinum, *shape)
    rvs = rvs / torch.norm(rvs.reshape((rvs.shape[0], rvs.shape[1], -1)), p=2, dim=-1).reshape((rvs.shape[0], rvs.shape[1], *[1]*len(FL_params.image_shape)))   #(tarnum, noinum, *shape)
    perturbeds = samples.unsqueeze(1) + (deltas.reshape((deltas.shape[0],*[1]*(len(rvs.shape)-1))) * rvs)     #(tarnum, noinum, *shape)
    perturbeds = perturbeds.reshape(-1, *FL_params.image_shape)   #(tarnum * noinum, *shape)
    decisions = decision_function(model, perturbeds, adv_labels, FL_params)         #(tarnum, noinum)
    decision_shape = [*decisions.shape] + [1] * len(FL_params.image_shape)  #(tarnum , noinum, [1]*len_shape)
    fvals = 2 * decisions.reshape(decision_shape) - 1.0   #(tarnum , noinum, [1]*len_shape)
    fvals -= torch.mean(fvals, dim=1).unsqueeze(1)
    gradfs = torch.mean(fvals * rvs, dim=1)   #(tarnum, *shape)
    return gradfs
def objective_function(model, x_advs, x_oris, adv_labels, distances, FL_params):
    x_subs = x_advs - x_oris   #(tarnum, *shape)
    x_subs = x_subs.view(x_subs.shape[0],-1)   #(tarnum,-1)
    x_subs = x_subs / torch.norm(x_subs, p=2, dim=-1).unsqueeze(1)  #(tarnum,-1)
    grad_xadvs = gradient_estimation_boundary(model, x_advs, adv_labels, distances, FL_params)  #(tarnum, *shape)
    grad_xadvs = grad_xadvs.view(grad_xadvs.shape[0], -1)
    grad_xadvs = grad_xadvs / torch.norm(grad_xadvs, p=2, dim=-1).unsqueeze(1)  #(tarnum,-1)
    cos = -torch.cosine_similarity(x_subs, grad_xadvs, dim=-1)  #(tarnum,)
    return cos
def adv_attack(model, x_samples, y_samples, tgt_samples, adv_labels, mode, FL_params):
    adv_inits = binary_search0(model, x_samples, tgt_samples, y_samples, adv_labels, FL_params)
#         adv_init = binary_search(model, x_sample, tgt_sample, adv_label, FL_params)
    adv_updates = adv_inits
    ori_samples = x_samples
    distance_inits = torch.norm((adv_inits - ori_samples).reshape((adv_inits.shape[0],-1)), p=2, dim=-1)   #(tarnum,)
    distance_values = distance_inits
    decay_epoch = 0
    mis_exist = False
    adv_lr = FL_params.adv_lr
    for adv_epoch in range(FL_params.adv_iter):
        adv_samples = copy.deepcopy(adv_updates)
        distances = copy.deepcopy(distance_values)
        decay_epoch +=1
        adv_samples.requires_grad = True
        losses = objective_function(model, adv_samples, ori_samples, adv_labels, distances, FL_params)  #(tarnum,)
        mis_nan = torch.nonzero(torch.isnan(losses), as_tuple=True)[0]
        not_nan = torch.nonzero(~torch.isnan(losses), as_tuple=True)[0]
        print("mis_nan: ", mis_nan)
        losses.sum().backward()
        grads = adv_samples.grad   #(tarnum, *shape)
        grads = grads / torch.norm(grads.reshape((grads.shape[0],-1)), p=2, dim=-1).reshape(grads.shape[0],*[1]*len(FL_params.image_shape))
        adv_samples.requires_grad = False           
        if decay_epoch % 10 == 0:
            adv_lr *= FL_params.decay_lr
        adv_samples[not_nan] -= adv_lr * grads[not_nan]
        
        beta = FL_params.beta_init
        while not torch.all(is_adversarial(model, adv_samples, adv_labels, FL_params)):
            is_adv = is_adversarial(model, adv_samples, adv_labels, FL_params)  #(tarnum,)
            not_adv = torch.nonzero(is_adv==0, as_tuple=True)[0]
            beta += FL_params.beta_step_size
            adv_samples[not_adv] += beta * (adv_samples[not_adv]-ori_samples[not_adv]) / \
                          torch.norm((adv_samples[not_adv]-ori_samples[not_adv]).reshape((len(not_adv),-1)), p=2, dim=-1).reshape((len(not_adv), *[1]*len(FL_params.image_shape)))
            if beta > FL_params.beta_max:
                mis_exist = True
                print('exist!')
                adv_samples[not_adv] = adv_updates[not_adv]
                print(not_adv)
                break
#         adv_samples[not_adv] = binary_search0(model, ori_samples[not_adv], adv_samples[not_adv], y_samples[not_adv], adv_labels[not_adv], FL_params)
        adv_samples = binary_search0(model, ori_samples, adv_samples, y_samples, adv_labels, FL_params)
#             adv_sample = binary_search(model, ori_sample, adv_sample, adv_label, FL_params)
        distance_values = torch.norm((adv_samples - ori_samples).reshape((adv_samples.shape[0], -1)), p=2, dim=-1)  #(tarnum,)
        adv_updates = copy.deepcopy(adv_samples)
        if FL_params.adv_with_test:
            losses_now = objective_function(model, adv_samples, ori_samples, adv_labels, distances, FL_params)
            for ide in range(len(adv_samples)):
                print("mode:{}, adv_update {}/{}: local_epoch={}/{}, adv_label={}, ori_label={}, loss={}, distance={}".format(mode, ide+1, FL_params.target_num,
                    adv_epoch+1, FL_params.adv_iter, torch.argmax(model(adv_samples[ide].unsqueeze(0).cuda())).cpu(), y_samples[ide], losses_now[ide], distance_values[ide]))
    if mis_exist is False:
        if len(mis_nan) != 0:
            not_adv = mis_nan
        else:
            not_adv = None
    else:
        not_adv = torch.cat((mis_nan, not_adv))
    return distance_values, not_adv
def robustness_estimation(model, x_data_set, y_data_set, tgt_list, mode, FL_params):
    Idl = torch.argsort(model(x_data_set.cuda()))[:,FL_params.class_num - 2].cpu()
    tgt_tgt = tgt_list[Idl]
    d_val, mis_adv = adv_attack(model, x_data_set, y_data_set, tgt_tgt, Idl, mode, FL_params)
    return d_val, mis_adv

'''
FL_params.adv_lr = 0.3
FL_params.decay_lr = 1
FL_params.target_num = 1
distance_train = torch.zeros((FL_params.sample_num, ))
distance_test = torch.zeros((FL_params.sample_num, ))
mis_adv_train = list()
mis_adv_test = list()
for idx in range(math.ceil(FL_params.sample_num / FL_params.target_num)):
    print('target iter: {}/{}'.format(idx+1, math.ceil(FL_params.sample_num / FL_params.target_num)))
    if idx == math.ceil(FL_params.sample_num / FL_params.target_num)-1:
        distance_train[idx*FL_params.target_num:FL_params.sample_num], not_adv_train = robustness_estimation(model, X_tgt_dataset[idx*FL_params.target_num:FL_params.sample_num], Y_tgt_dataset[idx*FL_params.target_num:FL_params.sample_num], tgt_list, 'train', FL_params)
        distance_test[idx*FL_params.target_num:FL_params.sample_num], not_adv_test = robustness_estimation(model, X_test_dataset[idx*FL_params.target_num:FL_params.sample_num], Y_test_dataset[idx*FL_params.target_num:FL_params.sample_num], tgt_list, 'test', FL_params)        
    else:
        distance_train[idx*FL_params.target_num:(idx+1)*FL_params.target_num], not_adv_train = robustness_estimation(model, X_tgt_dataset[idx*FL_params.target_num:(idx+1)*FL_params.target_num], Y_tgt_dataset[idx*FL_params.target_num:(idx+1)*FL_params.target_num], tgt_list, 'train', FL_params)
        distance_test[idx*FL_params.target_num:(idx+1)*FL_params.target_num], not_adv_test = robustness_estimation(model, X_test_dataset[idx*FL_params.target_num:(idx+1)*FL_params.target_num], Y_test_dataset[idx*FL_params.target_num:(idx+1)*FL_params.target_num], tgt_list, 'test', FL_params)
    if not_adv_train is not None:
        mis_adv_train.append(idx*FL_params.target_num+not_adv_train)
    if not_adv_test is not None:
        mis_adv_test.append(idx*FL_params.target_num+not_adv_test)
torch.save(distance_train, './tear_come/data/' + str(FL_params.data_name) + '/tear_advs_mem.pth')
torch.save(distance_test, './tear_come/data/' + str(FL_params.data_name) + '/tear_advs_nonmem.pth')
'''