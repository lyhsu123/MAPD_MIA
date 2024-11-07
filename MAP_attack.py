import torch.nn.functional as F
from tqdm import tqdm
import torch
def noise_generate(num=100, radius=0.5, fig_size=(3,32,32), random_seed=314):
    size = (num, *fig_size)
    torch.manual_seed(random_seed)
    noise = torch.randn(size)    #(noinum, *shape)
    noise_norm = torch.norm(noise.reshape((noise.shape[0],-1)), p=2, dim=-1)  #orch.Size(noinum)
    noise = noise / noise_norm.reshape((noise.shape[0],*([1]*len(fig_size)))) * radius
    return noise
def is_adversarials(model, perturbeds, adv_labels, FL_params):
    adv_labels = adv_labels.unsqueeze(1).repeat(1,perturbeds.shape[1])  #(samnum, noinum)
    predict_labels = torch.argmax(model(perturbeds.reshape((-1,*FL_params.image_shape)).cuda()), dim=-1).reshape(*(adv_labels.shape)).cpu()
    return predict_labels == adv_labels    #(samnum, noinum)
def is_adversarials_ori(model, perturbeds, ori_labels, FL_params):
    ori_labels = ori_labels.unsqueeze(1).repeat(1,perturbeds.shape[1])  #(samnum, noinum)
    predict_labels = torch.argmax(model(perturbeds.reshape((-1,*FL_params.image_shape)).cuda()), dim=-1).reshape(*(ori_labels.shape)).cpu()
    return predict_labels != ori_labels    #(samnum, noinum)
# def projected(oris, advs, x_advs, thetas):
#     is_adv = is_adversarials_ori()
#     thetas = thetas.reshape((*(thetas.shape),1,1,1))
#     return (1.0 - thetas) * oris + thetas * advs
    
def binary_search_circle(model, x_ori, x_adv, adv_o, adv_labels, ori_labels, FL_params, tol=0.00001):
    radius = torch.norm((x_adv-adv_o.unsqueeze(1)).reshape((x_adv.shape[0],x_adv.shape[1],-1)), p=2, dim=-1)   #(samnum, noinum)
    adv_dir = (x_ori - adv_o) / torch.norm((x_ori - adv_o).reshape((x_ori.shape[0],-1)), p=2, dim=-1).reshape((x_ori.shape[0],*([1]*len(FL_params.image_shape))))
    cln = adv_o.unsqueeze(1) + radius.reshape((*radius.shape,*([1]*len(FL_params.image_shape)))) * adv_dir.unsqueeze(1)    #(samnum, noinum, *shape)
    adv = adv_o.unsqueeze(1) - radius.reshape((*radius.shape,*([1]*len(FL_params.image_shape)))) * adv_dir.unsqueeze(1)    #(samnum, noinum, *shape)
    cln_labels = torch.argmax(model(cln.reshape((-1,*FL_params.image_shape)).cuda()), dim=-1).cpu()    #(samnum * noinum,)
    try:
        assert (cln_labels == ori_labels.repeat_interleave(cln.shape[1])).all(), "{}".format(cln_labels)
    except AssertionError as e:
        print('AssertionError: ', e)
    is_adv = is_adversarials_ori(model, x_adv, ori_labels, FL_params)    #(samnum, noinum)
    highs_ind = torch.where(is_adv, 1, 0).reshape((*is_adv.shape, *([1]*len(FL_params.image_shape))))   #(samnum, noinum, *[1]*shape_len)
    highs = highs_ind * x_adv + (1 - highs_ind) * adv     #(samnum, noinum, *shape)
    lows_ind = torch.where(is_adv, 1, 0).reshape((*is_adv.shape, *([1]*len(FL_params.image_shape)))) 
    lows = lows_ind * cln + (1 - lows_ind) * x_adv     #(samnum, noinum, *shape)
    old_mids = highs
    process = 0
    with tqdm(total = highs.shape[0] * highs.shape[1],desc='Progress',unit='units') as pbar:
        while torch.any(torch.norm((highs - lows).reshape((*is_adv.shape, -1)), p=2, dim=-1) > tol):
            mids = (highs + lows) / 2.0
            mid_purturbeds = (radius / torch.norm((mids-adv_o.unsqueeze(1)).reshape((*is_adv.shape,-1)),p=2,dim=-1)).reshape((*radius.shape,*([1]*len(FL_params.image_shape))))  \
                        * (mids-adv_o.unsqueeze(1)) + adv_o.unsqueeze(1)
            is_adv = is_adversarials_ori(model, mid_purturbeds, ori_labels, FL_params)
            highs_ind = torch.where(is_adv, 1, 0).reshape((*is_adv.shape,*([1]*len(FL_params.image_shape))))
            highs = highs_ind * mid_purturbeds + (1-highs_ind) * highs
            lows = highs_ind * lows + (1-highs_ind) * mid_purturbeds
            reached_numerical_precision = (old_mids == mid_purturbeds).all()
            process = torch.count_nonzero((torch.norm((highs - lows).reshape((*is_adv.shape, -1)), p=2, dim=-1) <= tol) + torch.all(old_mids.reshape((old_mids.shape[0],old_mids.shape[1],-1))==mid_purturbeds.reshape((old_mids.shape[0],old_mids.shape[1],-1)), dim=-1))
#             if process.item() >= 39995:
#                 sorts = torch.where((torch.norm(highs - lows, p=2, dim=(2,3,4)) <= tol).reshape(-1) + torch.all(old_mids.reshape((old_mids.shape[0],old_mids.shape[1],-1))==mid_purturbeds.reshape((old_mids.shape[0],old_mids.shape[1],-1)), dim=-1).reshape(-1)==0)
#                 print(sorts)
#                 print(torch.norm(highs - lows, p=2, dim=(2,3,4)).reshape(-1)[sorts])
            old_mids = mid_purturbeds
            pbar.update(process.item() - pbar.n)
            if reached_numerical_precision:
                break
    return highs                           #(samnum, noinum, *shape)

def projected_adv_attack(model, advs, oris, FL_params, noise_num=100, radius=0.5, random_seed=314, tol=0.00001, mode='binary'):
    adv_labels = torch.argmax(model(advs.cuda()), dim=-1).cpu()   #(samnum, )
    ori_labels = torch.argmax(model(oris.cuda()), dim=-1).cpu()   #(samnum, )
    noise = noise_generate(num=noise_num, radius=radius, fig_size=FL_params.image_shape, random_seed=random_seed)   #(noinum, *shape)
    advs_noise = advs.unsqueeze(1) + noise.unsqueeze(0)    #(samnum, noinum, *shape)
    adv_bounds = binary_search_circle(model, oris, advs_noise, advs, adv_labels, ori_labels, FL_params, tol=tol)
    return adv_bounds

def pro_dis_dir(projected_advs, X_dataset, mode, FL_params):
    pro_distances = torch.norm((projected_advs.cuda() - X_dataset.unsqueeze(1).cuda()).reshape((projected_advs.shape[0],projected_advs.shape[1], -1)), p=2, dim=-1)  #(samnum, noinum)
    pro_directions = (projected_advs.cuda() - X_dataset.unsqueeze(1).cuda()) / pro_distances.reshape((*pro_distances.shape,*([1]*len(FL_params.image_shape))))
    pro_distances_normv = torch.norm(pro_distances, p=2, dim=1)  #(samnum, )
    pro_distances_norm = pro_distances / pro_distances_normv.unsqueeze(1)   #(samnum, noinum)
    pro_distances_norm = torch.cat((pro_distances_norm,pro_distances_normv.unsqueeze(1)), dim=1)   #(samnum, noinum+1)
    pro_directions = pro_directions.reshape((pro_directions.shape[0],pro_directions.shape[1],-1))
    pro_directions_reverse = torch.cat((pro_directions[:,1:,:], pro_directions[:,0,:].unsqueeze(1)), dim=1)
    pro_directions_sim= F.cosine_similarity(pro_directions,pro_directions_reverse,dim=-1)
    print('pro_distances: ', pro_distances.shape)
    print('pro_distances_norm: ', pro_distances_norm.shape)
    print('pro_directions_sim: ', pro_directions_sim.shape)
    torch.save(pro_distances.cpu(), './data/' + str(FL_params.data_name) + '/pro_distances_{}.pth'.format(mode))
    torch.save(pro_distances_norm.cpu(), './data/' + str(FL_params.data_name) + '/pro_distances_norm_{}.pth'.format(mode))
    torch.save(pro_directions_sim.cpu(), './data/' + str(FL_params.data_name) + '/pro_directions_sim_{}.pth'.format(mode))