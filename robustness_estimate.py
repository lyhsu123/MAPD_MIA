import numpy as np
import torch
import math
import foolbox as fb
import foolbox.criteria as criteria
import math

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

def hsja(model, X_dataset, num, num_iter, Y_dataset, tgt_list, mode, FL_params):
    Adv_sample = torch.zeros((FL_params.class_num, num, *FL_params.image_shape), device=FL_params.device)
    for iters in range(math.ceil(num / num_iter)):
        for idl in range(FL_params.class_num-1):
            print("mode:{}, iters:{}/{}, label_idx:{}/{}".format(mode, iters+1, math.ceil(num / num_iter), idl, FL_params.class_num-1))
            if iters == math.ceil(num / num_iter)-1:
                adv_label = torch.arange(FL_params.class_num).repeat(num-(math.ceil(num / num_iter)-1)*num_iter,1)
            else:
                adv_label = torch.arange(FL_params.class_num).repeat(num_iter,1)
            mask = torch.ones_like(adv_label, dtype=torch.bool)
            if iters == math.ceil(num / num_iter)-1:
                mask[torch.arange(adv_label.shape[0]), Y_dataset[iters*num_iter:]] = False
            else:
                mask[torch.arange(adv_label.shape[0]), Y_dataset[iters*num_iter: (iters+1)*num_iter]] = False
            adv_label = adv_label[mask].reshape((adv_label.shape[0], -1)).long().cuda()
            print(adv_label)
            tgt_sample = tgt_list[adv_label[:,idl]]
            model.eval()
            fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
            attack = fb.attacks.HopSkipJump(init_attack=None)
            attack_criterion = criteria.TargetedMisclassification(adv_label[:,idl])
            if iters == math.ceil(num / num_iter)-1:
                x_advs = attack(fmodel, X_dataset[iters*num_iter:].cuda(), criterion=attack_criterion, starting_points=tgt_sample, epsilons=None)
                Adv_sample[idl, iters*num_iter:] = x_advs[0]
            else:
                x_advs = attack(fmodel, X_dataset[iters*num_iter: (iters+1)*num_iter].cuda(), criterion=attack_criterion, starting_points=tgt_sample, epsilons=None)
                Adv_sample[idl, iters*num_iter: (iters+1)*num_iter] = x_advs[0]
    torch.save(Adv_sample.cpu(), './data/' + str(FL_params.data_name) + '/hsja_adv_sample_{}+{}.pth'.format(mode, num))
    return Adv_sample.cpu()
