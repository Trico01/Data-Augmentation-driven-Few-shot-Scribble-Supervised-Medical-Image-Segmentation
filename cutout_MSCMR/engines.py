import math
import sys
import random
import time
import datetime
from typing import Iterable
import torch.nn.functional as Func
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from cutout import Cutout, rotate_invariant, rotate_back
from inference import keep_largest_connected_components

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def Cutmix_augment(x, l, device, beta=1):
    lams = []
    try:
        x=x.tensors
    except:
        pass
    lam = torch.distributions.beta.Beta(beta, beta).sample([x.shape[0], 1, 1, 1])
    bboxs = []
    x_flip = torch.flip(x,(0,))
    l_flip = torch.flip(l,(0,))
    for index in range(x.shape[0]):
        bbx1, bby1, bbx2, bby2= rand_bbox(x.shape, lam[index,0,0,0])
        x[index,:,bbx1:bbx2,bby1:bby2] = 0 #x_flip[index,:,bbx1:bbx2,bby1:bby2]
        l[index,:,bbx1:bbx2,bby1:bby2]= 0 #l_flip[index,:,bbx1:bbx2,bby1:bby2]
        bboxs.append([bbx1, bby1, bbx2, bby2])
    return x, l, bboxs

def Cutmix_targets(samples, targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    aug_samples, aug_targets, bboxs = Cutmix_augment(samples, target_masks, device)
    return aug_samples, aug_targets, bboxs

def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def mix_fixboxs(outputs, boxs, device):
    src_masks = outputs["pred_masks"]
    new_masks = torch.zeros_like(src_masks)
    masks_flip = torch.flip(src_masks,(0,))
    for index in range(src_masks.shape[0]):
        box = boxs[0]
        bbx1, bby1, bbx2, bby2 = box[0], box[1], box[2], box[3]
        new_masks[index,:,:,:] = src_masks[index,:,:,:]
        new_masks[index,:,bbx1:bbx2,bby1:bby2] = 0 #masks_flip[index,:,bbx1:bbx2,bby1:bby2]
    return new_masks

def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('loss_multiDice', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        # mix samples and targets
        samples_mixed, targets_mixed, bboxs_mixed = Cutmix_targets(samples, targets, device)
        # samples_cut, targets_cut, masks_cut = Cutout(samples_mixed, targets_mixed, device)
        # masks_cut = masks_cut.to(device)
        # samples_cut, targets_cut, angles = rotate_invariant(samples_cut, targets_cut)
        # outputs_cut = model(samples_cut, task)
        # samples_cut_back, outputs_cut,targets_cut = rotate_back(samples_cut, outputs_cut["pred_masks"],targets_cut,angles)
        outputs_mixed = model(samples_mixed, task)
        ###

        ## original
        # targets_onehot= convert_targets(targets,device)
        # outputs = model(samples.tensors, task)

        ## mix outputs
        # mixed_outputs = mix_fixboxs(outputs, bboxs_mixed, device)
        # mixed_outputs = mixed_outputs*masks_cut

        # original loss
        # loss_dict = criterion(outputs, targets_onehot)
        # weight_dict = criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in ['loss_CrossEntropy'])
        # if step == 0:
        #     print("original loss:", losses.item())

        # mixup loss
        loss_dict_mixed = criterion(outputs_mixed, targets_mixed)
        weight_dict = criterion.weight_dict
        losses_mixed = sum(loss_dict_mixed[k] * weight_dict[k] for k in loss_dict_mixed.keys() if k in ['loss_CrossEntropy'])
        if step == 0:
            print("mixup loss:", losses_mixed.item())

        # integrity loss
        # original_masks = outputs["pred_masks"]
        # predictions_original_list = []
        
        # for i in range(original_masks.shape[0]):
        #     prediction = np.uint8(np.argmax(original_masks[i,:,:,:].detach().cpu(), axis=0))
        #     prediction = keep_largest_connected_components(prediction)
        #     prediction = torch.from_numpy(prediction).to(device)
        #     predictions_original_list.append(prediction)

        # predictions = torch.stack(predictions_original_list)
        # predictions = torch.unsqueeze(predictions, 1)
        # prediction_onehot = to_onehot_dim4(predictions,device)
        
        # loss_dict_integrity = criterion({"pred_masks": prediction_onehot}, targets_onehot)
        # losses_integrity = sum(loss_dict_integrity[k] * weight_dict[k] for k in loss_dict_integrity.keys() if k in ['loss_CrossEntropy'])
        # if step == 0:
        #     print("integrity loss:", losses_integrity.item())

        # invariant loss
        # invariant_loss = 1- Func.cosine_similarity(outputs_cut["pred_masks"], mixed_outputs, dim=1).mean()
        # invariant_loss = 0.05*invariant_loss
        # if step == 0:
        #     print("invariant loss:", invariant_loss.item())

        final_losses = losses_mixed

        optimizer.zero_grad()
        final_losses.backward()
        optimizer.step()

        loss_dict_reduced = utils.reduce_dict(loss_dict_mixed)
        loss_dict_reduced_unscaled = { f'{k}_unscaled': v for k, v in loss_dict_reduced.items() }
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy']}

        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, dataloader_dict, device, output_dir, visualizer, epoch, writer):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('loss_multiDice', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ] 
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        targets_onehot= convert_targets(targets,device)
        outputs = model(samples.tensors, task)

        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict.keys()}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 16.) == 0:  
            ##original  
            sample_list.append(samples.tensors[0])
            ##
            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            
            ##original
            target_list.append(targets[0]['masks'])
            ##
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    writer.add_scalar('avg_loss', stats['loss_CrossEntropy'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)
    
    return stats