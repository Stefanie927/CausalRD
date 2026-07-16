# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import pandas as pd
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from util.datasets import TestDataset_genCounterfactual
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from diffusion_utils.templates import *
from renderer import *

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,  # 8
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,   # causal model
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-5, metavar='LR',  # 5e-3
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='./pth/RETFound_cfp_weights.pth',type=str,
                        help='finetune from checkpoint')   # ./pth/RETFound_cfp_weights.pth
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_true', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/jupyter/Mor_DR_data/data/data/IDRID/Disease_Grading/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir/causalRD_gen_counterfactual/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/causalRD_gen_counterfactual/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)   # 4 10
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # causaldiffusionAE
    parser.add_argument('--diffusion_lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    # parser.add_argument('--image_size', default=96, type=int)   # 96
    parser.add_argument('--image_size', default=224, type=int)    # 128
    parser.add_argument('--save_interval', default=10000, type=int)  # 每10000个step保存一次
    parser.add_argument('--max_step', default=900000, type=int)  
    

    return parser


def main(args):
    misc.init_distributed_mode(args)
    conf = odir_autoenc()

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    cudnn.benchmark = True
    
    test_df = pd.read_csv('./data/OIA_ODIR/off-site_test.csv')
    test_img_dir = r"./data/OIA_ODIR/cropped_ODIR-5K_Offsite_Testing_Images"
    test_encoder_feature_pkl = r"./data/OIA_ODIR/ViT_features/test_features.pkl"

    dataset_test = TestDataset_genCounterfactual(test_df, test_encoder_feature_pkl, test_img_dir, is_train=False, args=args)

    data_loader_test = DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False)
    
    model = conf.make_model_conf().make_model()
    model = model.to(device)

    # load causal+diffusion
    ckpt_path = r"./ckpt/OIA_ODIR/opt620000_diffusion.pth"
    checkpoint = torch.load(ckpt_path, map_location=device) 
    state_dict = checkpoint['model_state_dict']
    model_sd_keys = set(model.state_dict().keys())
    ckpt_sd_keys = set(state_dict.keys())
    missing_keys = model_sd_keys - ckpt_sd_keys
    unexpected_keys = ckpt_sd_keys - model_sd_keys
    print(f"diffusion Missing keys: {missing_keys}")
    print(f"diffusion Unexpected keys: {unexpected_keys}")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    causal_A = model.causal_model.state_dict()['DisentanglementEncoder.A']
    causal_cdag = model.causal_model.state_dict()['DisentanglementEncoder.causal_dag']

    DR_cls = nn.Sequential(
        nn.Linear(256, 1024),
        nn.Linear(1024, 1)
        )
    G_cls = nn.Sequential(
        nn.Linear(256, 1024),
        nn.Linear(1024, 1)
        )
    C_cls =  nn.Sequential(
        nn.Linear(256, 1024),
        nn.Linear(1024, 1)
        )
    H_cls =  nn.Sequential(
        nn.Linear(256, 1024),
        nn.Linear(1024, 1)
        )

    # load
    DR_cls.load_state_dict(torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_D_checkpoint.pth", map_location=device, weights_only=False)['model_state_dict'], strict=False)
    G_cls.load_state_dict(torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_G_checkpoint.pth", map_location=device, weights_only=False)['model_state_dict'], strict=False)
    C_cls.load_state_dict(torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_C_checkpoint.pth", map_location=device, weights_only=False)['model_state_dict'], strict=False)
    H_cls.load_state_dict(torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_H_checkpoint.pth", map_location=device, weights_only=False)['model_state_dict'], strict=False)

    DR_cls = DR_cls.to(device)
    G_cls = G_cls.to(device)
    C_cls = C_cls.to(device)
    H_cls = H_cls.to(device)

    DR_cond_dict = torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_D_mean_std.pt") 
    G_cond_dict = torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_G_mean_std.pt")
    C_cond_dict = torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_C_mean_std.pt")
    H_cond_dict = torch.load(r"./ckpt/OIA_ODIR/counterfactual_cls/concept_H_mean_std.pt")

    DR_weight = 0.5  # 0.5, 0.8, 1
    G_weight = 1.5  # 1.8, 2
    C_weight = 1.5  # 2  
    H_weight = 2.5  # 2

    for image_diff, encoder_feat, image_id, targets, image_name  in data_loader_test:

        image_diff = image_diff.to(device, non_blocking=True)
        encoder_feat = encoder_feat.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        cond_feat = model.causal_model.get_single_features(encoder_feat, targets)
        cond_feat = cond_feat.view(cond_feat.shape[0], -1)  # flatten
        cond = cond_feat.clone()
        xT = encode_stochastic(model, image_diff, conf, cond, T=250)

        ### DR ###
        cond = cond_feat.clone()
        cond_DR = (cond[:, 512:768] - DR_cond_dict["conds_mean"].to(device)) / DR_cond_dict["conds_std"].to(device)   # normalize
        cond_DR = cond_DR + DR_weight * math.sqrt(256) * F.normalize(torch.matmul(DR_cls[1].weight, DR_cls[0].weight), dim=1)   
        cond_DR = (cond_DR * DR_cond_dict["conds_std"].to(device)) + DR_cond_dict["conds_mean"].to(device)  # denormalize
        cond[:, 512:768] = cond_DR
        
        cond_DR_A = causality_based_concept_emb(cond, causal_A, device)
        sample_DR = render(model, xT, conf, cond_DR_A, T=100)
        for ix in range(sample_DR.shape[0]):
            save_image(sample_DR[ix], rf"gen_DR_{image_name[ix]}_2.jpg")

        ### G ###
        cond = cond_feat.clone()
        cond_G = (cond[:,768:1024] - G_cond_dict["conds_mean"].to(device)) / G_cond_dict["conds_std"].to(device)   # normalize
        cond_G = cond_G + G_weight * math.sqrt(256) * F.normalize(torch.matmul(G_cls[1].weight, G_cls[0].weight), dim=1)   
        cond_G = (cond_G * G_cond_dict["conds_std"].to(device)) + G_cond_dict["conds_mean"].to(device)  # denormalize
        cond[:, 768:1024] = cond_G

        cond_G_A = causality_based_concept_emb(cond, causal_A, device)
        sample_G = render(model, xT, conf, cond_G_A, T=100)
        for ix in range(sample_G.shape[0]):
            save_image(sample_G[ix], rf"gen_G_{image_name[ix]}.jpg")

        ### C ###
        cond = cond_feat.clone()
        cond_C = (cond[:,1024:1280] - C_cond_dict["conds_mean"].to(device)) / C_cond_dict["conds_std"].to(device)   # normalize
        cond_C = cond_C + C_weight * math.sqrt(256) * F.normalize(torch.matmul(C_cls[1].weight, C_cls[0].weight), dim=1)   # cls的weight也是1792 
        cond_C = (cond_C * C_cond_dict["conds_std"].to(device)) + C_cond_dict["conds_mean"].to(device)  # denormalize
        cond[:, 1024:1280] = cond_C
        cond_C_A = causality_based_concept_emb(cond, causal_A, device)
        sample_C = render(model, xT, conf, cond_C_A, T=100)
        for ix in range(sample_C.shape[0]):
            save_image(sample_C[ix], rf"gen_C_{image_name[ix]}.jpg")

        ### H ###
        cond = cond_feat.clone()
        cond_H = (cond[:, 1280:1536] - H_cond_dict["conds_mean"].to(device)) / H_cond_dict["conds_std"].to(device)   # normalize
        cond_H = cond_H + H_weight * math.sqrt(256) * F.normalize(torch.matmul(H_cls[1].weight, H_cls[0].weight), dim=1)   # cls的weight也是1792 
        cond_H = (cond_H * H_cond_dict["conds_std"].to(device)) + H_cond_dict["conds_mean"].to(device)  # denormalize
        cond[:, 1280:1536] = cond_H
        cond_H_A = causality_based_concept_emb(cond, causal_A, device)
        sample_H = render(model, xT, conf, cond_H_A, T=100)
        for ix in range(sample_H.shape[0]):
            save_image(sample_H[ix], rf"gen_H_{image_name[ix]}.jpg")




def causality_based_concept_emb(exo_concept_embs, causal_cdag, device):  # (I-A)^-1*e  
    exo_concept_embs = exo_concept_embs.reshape(-1,7,256)
    z = np.matmul(np.linalg.inv(np.eye(len(causal_cdag.cpu())) - np.array(causal_cdag.cpu())), np.array(exo_concept_embs.detach().cpu()))
    z = z.reshape(z.shape[0], -1)
    # return z.astype(np.float32)
    return torch.Tensor(z).to(device)



def encode_stochastic(model, x, conf, cond, T=None):
    if T is None:
        sampler = conf.make_eval_diffusion_conf().make_sampler()
    else:
        sampler = conf._make_diffusion_conf(T).make_sampler()
    out = sampler.ddim_reverse_sample_loop(model,
                                            x,
                                            model_kwargs={'cond': cond})
    return out['sample']


def render(model, noise, conf, cond=None, T=None):
    if T is None:
        sampler = conf.make_eval_diffusion_conf().make_sampler()
    else:
        sampler = conf._make_diffusion_conf(T).make_sampler()

    pred_img = render_condition(conf,
                                model,
                                noise,
                                sampler=sampler,
                                cond=cond)

    # pred_img = (pred_img + 1) / 2  

    return pred_img


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
