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
from mixup import mixup_process, get_lambda
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from cutout import Cutout


class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()
        
    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
        
    def forward(self, originals, inputs, outputs, ori_labels, labels, epoch, writer):
        self.save_image(originals, 'inputs_original', epoch, writer)
        self.save_image(inputs, 'inputs_train', epoch, writer)
        self.save_image(outputs.float(), 'outputs_train', epoch, writer)
        self.save_image(ori_labels.float(), 'labels_original', epoch, writer)
        self.save_image(labels.float(), 'labels_train', epoch, writer)


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

def to_onehot(target_masks, device):
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
                    device: torch.device, epoch: int, args, writer):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    original_list,sample_list, output_list, target_list, target_ori_list =[], [], [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        ##Cutout
        samples_cut, targets_cut = Cutout(samples, targets)
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        targets_onehot = convert_targets(targets,device)
        #puzzlemix
        samples_var = Variable(samples.tensors, requires_grad=True)
 
        ###
        model.eval()
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        losses.backward(retain_graph=True)

        ### original output:
        # output_original = model(samples_var,task)

        #puzzlemix
        unary = torch.sqrt(torch.mean(samples_var.grad **2, dim=1))
        unary = F.pad(unary, (22,22,22,22,0,0), 'constant')
        samples_var_256 = F.pad(samples_var, (22,22,22,22,0,0,0,0), 'constant')
        targets_onehot_256 = F.pad(targets_onehot, (22,22,22,22,0,0,0,0), 'constant')
        out, reweighted_target, indices_transport, mask_transport = mixup_process(samples_var_256, targets_onehot_256, args=args, grad= unary)
        out = out[:,:,22:-22,22:-22]
        reweighted_target = reweighted_target[:,:,22:-22,22:-22]
        mask_transport = mask_transport[:,:,22:-22,22:-22]
        # reweighted_target = reweighted_target.argmax(1,keepdim=True)
        # reweighted_target = to_onehot(reweighted_target, device)
        out = Variable(out)
        reweighted_target = Variable(reweighted_target)
        model.train()
        outputs_mix=model(out,task)
        ###

        ### mixed output
        # mixed_output = torch.zeros_like(outputs_mix["pred_masks"])
        # shuffled_output = output_original["pred_masks"][indices_transport].clone()
        # for i in range(shuffled_output.shape[1]):
        #     mixed_output[:,i,:,:] = output_original["pred_masks"][:,i,:,:] * mask_transport[:,0,:,:] + shuffled_output[:,i,:,:] * (1-mask_transport[:,0,:,:])

        # if step % 200 == 0:
        #     for i in range(samples_var.shape[0]):  
        #         original_list.append(samples_var[i])
        #         sample_list.append(out[i])
        #         _, pre_masks = torch.max(outputs_mix['pred_masks'][i], 0, keepdims=True)
        #         output_list.append(pre_masks)
        #         target_ori_list.append(targets_onehot.argmax(1,keepdim=True)[i])
        #         target_list.append(reweighted_target.argmax(1,keepdim=True)[i])

        # puzzlemix_loss
        loss_dict = criterion(outputs_mix, reweighted_target)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        if step == 0:
            print("puzzlemix loss:", losses.item())

        # original_loss
        # loss_dict_ori = criterion(output_original, targets_onehot)
        # weight_dict = criterion.weight_dict
        # losses_ori = sum(loss_dict_ori[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        # if step == 0:
        #     print("original loss:", losses_ori.item())

        # invariant_loss
        # invariant_loss = 1 - Func.cosine_similarity(outputs_mix["pred_masks"], mixed_output, dim=1).mean()
        # invariant_loss = 0.05*invariant_loss
        # if step == 0:
        #     print("invariant loss:", invariant_loss.item())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = { f'{k}_unscaled': v for k, v in loss_dict_reduced.items() }
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy']}
        optimizer.zero_grad()

        losses_final = losses #+ invariant_loss+losses_ori
        losses_final.backward()

        optimizer.step()
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

    # visual_train = Visualize_train()
    # visual_train(torch.stack(original_list),torch.stack(sample_list), torch.stack(output_list), torch.stack(target_ori_list),torch.stack(target_list), epoch, writer)
    
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

        #puzzlemix
        samples_var = Variable(samples.tensors, requires_grad=True)
        ###
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        #metric_logger.update(loss_multiDice=loss_dict_reduced['loss_multiDice'])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 16.) == 0:  
            ##original  
            # sample_list.append(samples.tensors[0])
            ##
            ##mixup
            sample_list.append(samples_var[0])
            ##

            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            
            ##mixup
            target_list.append(targets_onehot.argmax(1,keepdim=True)[0])
            ##
            ##original
            # target_list.append(targets[0]['masks'])
            ##
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    writer.add_scalar('loss_CrossEntropy', stats['loss_CrossEntropy'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)
    
    return stats

def infer(model, criterion, postprocessors, dataloader_dict, device, output_dir, visualizer, writer):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('loss_multiDice', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('loss_RV', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('loss_MYO', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('loss_LV', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    # loss_recorded = []
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
        # samples_mixed, targets_mixed, _ = mix_targets(samples, targets, device)
        # outputs = model(samples_mixed, task)
        ###

        outputs = model(samples, task)
        loss_dict = criterion(outputs, targets_mixed)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled)
        print(loss_dict_reduced.keys())
        metric_logger.update(loss_multiDice=loss_dict_reduced['loss_multiDice'])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 16.) == 0:    
            sample_list.append(samples.tensors[0])
            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            target_list.append(targets[0]['masks'])
            # target_list.append(targets.argmax(1,keepdim=True)[0])
    # print("length1:",len(loss_recorded), "length2:",len(loss_recorded[0]))
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    epoch = 0
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)
    
    return stats