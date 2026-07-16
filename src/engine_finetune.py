# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json



def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_


def load_data(data_loader):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    
    while True:
        yield from data_loader


def train_one_epoch(model: torch.nn.Module, 
                    # causal_model: torch.nn.Module, 
                    sampler,  # diffusion
                    T_sampler,
                    criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    # epoch: int, 
                    loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # data_loader_iter = iter(data_loader)
    data_loader_iter = load_data(data_loader)
    
    # print(causal_model.DisentanglementEncoder.A)
    step = 0
    while step < args.max_step:
        if step % 100 == 0:
            print (f"step: {step}")
        
        image_diff, encoder_feat, targets = next(data_loader_iter)  # encoder_feat.shape (bs, 1024)

        # we use a per iteration (instead of per epoch) lr scheduler
        # for causal model
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        image_diff = image_diff.to(device, non_blocking=True)
        encoder_feat = encoder_feat.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        # with torch.cuda.amp.autocast():
        t, _ = T_sampler.sample(image_diff.shape[0], torch.device("cuda:0")) 
        
        model_kwargs = {}
        model_kwargs['label'] = targets
        model_kwargs['encoder_feat'] = encoder_feat
        diffusion_loss = sampler.training_losses(model=model, x_start=image_diff, t=t, model_kwargs=model_kwargs)
        diffusion_mse_loss = diffusion_loss["loss"].mean()
        causal_loss = diffusion_loss["causal_loss"]

        loss = diffusion_mse_loss + causal_loss*0.0001
        loss_value = loss.item()

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(step + 1) % accum_iter == 0)

        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(diffusion_mse_loss=diffusion_mse_loss.item())
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)


        metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        if step < 200000:
            if step % (args.save_interval*2)==0:
                checkpoint = {
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                checkpoint_path = os.path.join(args.output_dir, f"opt{(step):06d}.pth")
                torch.save(checkpoint, checkpoint_path)
                print("Diffusion Model checkpoint saved at", checkpoint_path)

        if step >= 200000:
            if step % args.save_interval==0:
                checkpoint = {
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                checkpoint_path = os.path.join(args.output_dir, f"opt{(step):06d}.pth")
                torch.save(checkpoint, checkpoint_path)
                print("Diffusion Model checkpoint saved at", checkpoint_path)
                
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'step': step}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        step += 1



@torch.no_grad()
def evaluate_old(data_loader, model, device, task, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc




#calculate kappa, F-1 socre and AUC value
def ODIR_Metrics(gt_data, pr_data):
    """ function from ODIR2019 challenge """
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0

    threshold = 0.5  
    pr_data_binary = (pr_data >= threshold).astype(int)

    precisions = metrics.precision_score(gt_data, pr_data_binary, average=None)
    recalls = metrics.recall_score(gt_data, pr_data_binary, average=None)
    classes_f1 = metrics.f1_score(gt_data, pr_data_binary, average=None)
    for class_index in range(len(precisions)):
        print(f'Class {class_index + 1}: F1 = {classes_f1[class_index]}, Precision = {precisions[class_index]}, Recall = {recalls[class_index]}')

    return kappa, f1, auc, final_score




@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    val_loss = 0
    with torch.no_grad():
        for image_feat, image, labels in dataloader:
            image_feat = image_feat.to(device)
            labels = labels.to(device)

            concept_emb, outputs, output_age, output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = model(image_feat, labels)
            output = outputs
            loss = pred_o_loss + discriminator_loss + prior_kl + mask_recon_loss + feat_kl + mask_kl + label_mse
            val_loss += loss.item()
            all_labels.append(labels[:,2:].cpu().numpy())   
            all_probs.append(output.detach().cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)  

    kappa, f1, auc, final_score = ODIR_Metrics(all_labels, all_probs)  

    return val_loss / len(dataloader), final_score, kappa, f1, auc




@torch.no_grad()
def prediction(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for image_feat, image, img_id, labels in dataloader:
            image_feat = image_feat.to(device)
            labels = labels.to(device)

            concept_emb, outputs, output_age, output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = model(image_feat, labels)
            output = outputs
            all_labels.append(labels[:,2:].cpu().numpy())   
            all_probs.append(output.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs) 

    kappa, f1, auc, final_score = ODIR_Metrics(all_labels, all_probs)   

    return final_score, kappa, f1, auc