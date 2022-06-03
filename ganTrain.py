import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import float32, optim
from torch.utils import data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np

import os
import json
import random
import socket
from tqdm import tqdm
from argparse import ArgumentParser

from PIL import Image

import utils
import network
import datasets as dt
from metrics import StreamSegMetrics
from utils import ext_transforms as et
from utils import histeq as hq


def get_dataset(opts):

    mean = [0.485, 0.456, 0.406] if opts.is_rgb else [0.485]
    std = [0.229, 0.224, 0.225] if opts.is_rgb else [0.229]

    train_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
        #et.GaussianBlur(kernel_size=(5, 5)),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtRandomVerticalFlip(),
        et.ExtToTensor(),
        et.GaussianPerturb(),
        et.ExtNormalize(mean=mean, std=std),
        ])
    val_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.val_crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std)
        ])

    if opts.dataset == "CPN":
        train_dst = dt.CPNSegmentation(root=opts.data_root, datatype=opts.dataset,
                                        image_set='train', transform=train_transform, 
                                        is_rgb=opts.is_rgb)
        val_dst = dt.CPNSegmentation(root=opts.data_root, datatype=opts.dataset,
                                     image_set='val', transform=val_transform, 
                                     is_rgb=opts.is_rgb)
    elif opts.dataset == "CPN_all":
        train_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype=opts.dataset,
                                         image_set='train', transform=train_transform, 
                                         is_rgb=opts.is_rgb)
        val_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype=opts.dataset,
                                        image_set='val', transform=val_transform, 
                                        is_rgb=opts.is_rgb)
    elif opts.dataset == "Median":
        train_dst = dt.Median(root=opts.data_root, datatype=opts.dataset, 
                            image_set='train', transform=train_transform,
                            is_rgb=opts.is_rgb)
        val_dst = dt.Median(root=opts.data_root, datatype=opts.dataset,
                        image_set='val', transform=val_transform,
                        is_rgb=opts.is_rgb)
    elif opts.dataset == "CPN_aug":
        train_dst = dt.CPNaug(root=opts.data_root, datatype=opts.dataset, 
                            image_set='train', transform=train_transform,
                            is_rgb=opts.is_rgb)
        val_dst = dt.CPNaug(root=opts.data_root, datatype=opts.dataset,
                        image_set='val', transform=val_transform,
                        is_rgb=opts.is_rgb)
    elif opts.dataset == "CPN_all_ver01":
        train_dst = dt.CPNver(root=opts.data_root, datatype=opts.dataset, 
                            image_set='train', transform=train_transform,
                            is_rgb=opts.is_rgb)
        val_dst = dt.CPNver(root=opts.data_root, datatype=opts.dataset,
                        image_set='val', transform=val_transform,
                        is_rgb=opts.is_rgb)
    else:
        train_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset, 
                            image_set='train', transform=train_transform,
                            is_rgb=opts.is_rgb)
        val_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset,
                        image_set='val', transform=val_transform,
                        is_rgb=opts.is_rgb)
    
    return train_dst, val_dst

def build_log(opts, LOGDIR) -> SummaryWriter:
    # Tensorboard option
    if opts.save_log:
        logdir = os.path.join(LOGDIR, 'log')
        writer = SummaryWriter(log_dir=logdir)

    # Validate option
    if opts.val_results:
        logdir = os.path.join(LOGDIR, 'val_results')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        opts.val_results_dir = logdir

    # Save best model option
    if opts.save_model:
        logdir = os.path.join(LOGDIR, 'best_param')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        opts.save_ckpt = logdir
    else:
        logdir = os.path.join(LOGDIR, 'cache_param')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        opts.save_ckpt = logdir

    # Save Options description
    jsummary = {}
    for key, val in vars(opts).items():
        jsummary[key] = val
    utils.save_dict_to_json(jsummary, os.path.join(LOGDIR, 'summary.json'))

    return writer

def validate(opts, s_model, t_model, loader, device, metrics, epoch, criterion):

    metrics.reset()
    ret_samples = []

    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            s_outputs = s_model(images)
            probs = nn.Softmax(dim=1)(s_outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = labels.detach().cpu().numpy()

            t_outputs = t_model(images)

            loss = criterion(s_outputs, t_outputs, labels)

            metrics.update(target, preds)
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss

def load_model(opts: ArgumentParser = None, model_name: str = '', msg: str = '', 
                verbose: bool = False, pretrain: str = None, 
                output_stride: int = 8, sep_conv: bool = False):
    
    print("<load model>", msg) if verbose else 0

    try:    
        if model_name.startswith("deeplab"):
            model = network.model.__dict__[model_name](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes, output_stride=output_stride)
            if sep_conv and 'plus' in model_name:
                network.convert_to_separable_conv(model.classifier)
            utils.set_bn_momentum(model.backbone, momentum=0.01)
        else:
            model = network.model.__dict__[model_name](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes)
    except:
        raise Exception("<load model> Error occured while loading a model.")

    if pretrain is not None and os.path.isfile(pretrain):
        print("<load model> restored parameters from %s" % pretrain)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        checkpoint = torch.load(pretrain, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint  # free memory
        torch.cuda.empty_cache()

    return model
    
def train(opts, devices, LOGDIR) -> dict:

    writer = build_log(opts, LOGDIR)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ''' (1) Get datasets
    '''
    train_dst, val_dst = get_dataset(opts)
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size,
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    print("Dataset: %s, Train set: %d, Val set: %d" % 
                    (opts.dataset, len(train_dst), len(val_dst)))

    ''' (2) Set up criterion
    '''
    if opts.loss_type == 'kd_loss':
        criterion = None
    else:
        raise NotImplementedError

    ''' (3 -1) Load teacher & student models
    '''
    t_model = load_model(opts=opts, model_name=opts.t_model, verbose=True,
                            pretrain=opts.t_model_params,
                            msg=" Teacher model selection: {}".format(opts.t_model),
                            output_stride=opts.t_output_stride, sep_conv=opts.t_separable_conv).to(devices)
    
    s_model = load_model(opts=opts, model_name=opts.s_model, verbose=True,
                            msg=" Student model selection: {}".format(opts.s_model),
                            output_stride=opts.output_stride, sep_conv=opts.separable_conv)

    ''' (4) Set up optimizer
    '''
    if opts.s_model.startswith("deeplab"):
        if opts.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "Adam":
            optimizer = torch.optim.Adam(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise NotImplementedError
    else:
        optimizer = optim.RMSprop(s_model.parameters(), 
                                    lr=opts.lr, 
                                    weight_decay=opts.weight_decay,
                                    momentum=opts.momentum)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    ''' (5) Resume student model & scheduler
    '''
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        if torch.cuda.device_count() > 1:
            s_model = nn.DataParallel(s_model)
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        s_model.load_state_dict(checkpoint["model_state"])
        s_model.to(devices)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            resume_epoch = checkpoint["cur_itrs"]
            print("Training state restored from %s" % opts.ckpt)
        else:
            resume_epoch = 0
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
        torch.cuda.empty_cache()
    else:
        print("[!] Train from scratch...")
        resume_epoch = 0
        if torch.cuda.device_count() > 1:
            s_model = nn.DataParallel(s_model)
        s_model.to(devices)

    ''' (6) Set up metrics
    '''
    metrics = StreamSegMetrics(opts.num_classes)
    early_stopping = utils.EarlyStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.save_ckpt, save_model=opts.save_model)
    dice_stopping = utils.DiceStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.save_ckpt, save_model=opts.save_model)
    best_score = 0.0

    ''' (7) Train
    '''
    B_epoch = 0
    B_val_score = None

    for epoch in range(resume_epoch, opts.total_itrs):

        s_model.train()
        running_loss = 0.0
        metrics.reset()

        for (images, lbl) in tqdm(train_loader, leave=True):

            images = images.to(devices)
            lbl = lbl.to(devices)
            
            optimizer.zero_grad()

            s_outputs = s_model(images)
            probs = nn.Softmax(dim=1)(s_outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()

            t_outputs = t_model(images)

            weights = lbl.detach().cpu().numpy().sum() / (lbl.shape[0] * lbl.shape[1] * lbl.shape[2])
            weights = torch.tensor([weights, 1-weights], dtype=float32).to(devices)
            criterion = utils.KDLoss(weight=weights, alpha=opts.alpha, temperature=opts.T)
            loss = criterion(s_outputs, t_outputs, lbl)

            loss.backward()
            optimizer.step()
            metrics.update(lbl.detach().cpu().numpy(), preds)
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        score = metrics.get_results()

        epoch_loss = running_loss / len(train_loader.dataset)
        print("[{}] Epoch: {}/{} Loss: {:.8f}".format(
            'Train', epoch+1, opts.total_itrs, epoch_loss))
        print(" Overall Acc: {:.2f}, Mean Acc: {:.2f}, FreqW Acc: {:.2f}, Mean IoU: {:.2f}, Class IoU [0]: {:.2f} [1]: {:.2f}".format(
            score['Overall Acc'], score['Mean Acc'], score['FreqW Acc'], score['Mean IoU'], score['Class IoU'][0], score['Class IoU'][1]))
        print(" F1 [0]: {:.2f} [1]: {:.2f}".format(score['Class F1'][0], score['Class F1'][1]))
        
        if opts.save_log:
            writer.add_scalar('Overall_Acc/train', score['Overall Acc'], epoch)
            writer.add_scalar('Mean_Acc/train', score['Mean Acc'], epoch)
            writer.add_scalar('FreqW_Acc/train', score['FreqW Acc'], epoch)
            writer.add_scalar('Mean_IoU/train', score['Mean IoU'], epoch)
            writer.add_scalar('Class_IoU_0/train', score['Class IoU'][0], epoch)
            writer.add_scalar('Class_IoU_1/train', score['Class IoU'][1], epoch)
            writer.add_scalar('Class_F1_0/train', score['Class F1'][0], epoch)
            writer.add_scalar('Class_F1_1/train', score['Class F1'][1], epoch)
            writer.add_scalar('epoch_loss/train', epoch_loss, epoch)

        if (epoch+1) % opts.val_interval == 0:
            s_model.eval()
            metrics.reset()
            val_score, val_loss = validate(opts, s_model, t_model, val_loader, 
                                            devices, metrics, epoch, criterion)

            print("[{}] Epoch: {}/{} Loss: {:.8f}".format('Validate', epoch+1, opts.total_itrs, val_loss))
            print(" Overall Acc: {:.2f}, Mean Acc: {:.2f}, FreqW Acc: {:.2f}, Mean IoU: {:.2f}".format(
                val_score['Overall Acc'], val_score['Mean Acc'], val_score['FreqW Acc'], val_score['Mean IoU']))
            print(" Class IoU [0]: {:.2f} [1]: {:.2f}".format(val_score['Class IoU'][0], val_score['Class IoU'][1]))
            print(" F1 [0]: {:.2f} [1]: {:.2f}".format(val_score['Class F1'][0], val_score['Class F1'][1]))
            
            if early_stopping(val_loss, s_model, optimizer, scheduler, epoch):
                B_epoch = epoch
            if dice_stopping(-1 * val_score['Class F1'][1], s_model, optimizer, scheduler, epoch):
                B_val_score = val_score

            if opts.save_log:
                writer.add_scalar('Overall_Acc/val', val_score['Overall Acc'], epoch)
                writer.add_scalar('Mean_Acc/val', val_score['Mean Acc'], epoch)
                writer.add_scalar('FreqW_Acc/val', val_score['FreqW Acc'], epoch)
                writer.add_scalar('Mean_IoU/val', val_score['Mean IoU'], epoch)
                writer.add_scalar('Class_IoU_0/val', val_score['Class IoU'][0], epoch)
                writer.add_scalar('Class_IoU_1/val', val_score['Class IoU'][1], epoch)
                writer.add_scalar('Class_F1_0/val', val_score['Class F1'][0], epoch)
                writer.add_scalar('Class_F1_1/val', val_score['Class F1'][1], epoch)
                writer.add_scalar('epoch_loss/val', val_loss, epoch)
        
        if early_stopping.early_stop:
            print("Early Stop !!!")
            break

        if opts.run_demo and epoch > 3:
            print("Run demo !!!")
            break

    if opts.val_results:
        params = utils.Params(json_path=os.path.join(LOGDIR, 'summary.json')).dict
        for k, v in B_val_score.items():
            params[k] = v
        utils.save_dict_to_json(d=params, json_path=os.path.join(LOGDIR, 'summary.json'))

        if opts.save_model:
            checkpoint = torch.load(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt'), map_location=devices)
            s_model.load_state_dict(checkpoint["model_state"])
            sdir = os.path.join(opts.val_results_dir, 'epoch_{}'.format(B_epoch))
            utils.save(sdir, s_model, val_loader, devices, opts.is_rgb)
            del checkpoint
            del s_model
            torch.cuda.empty_cache()
        else:
            checkpoint = torch.load(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt'), map_location=devices)
            s_model.load_state_dict(checkpoint["model_state"])
            sdir = os.path.join(opts.val_results_dir, 'epoch_{}'.format(B_epoch))
            utils.save(sdir, s_model, val_loader, devices, opts.is_rgb)
            del checkpoint
            del s_model
            torch.cuda.empty_cache()
            if os.path.exists(os.path.join(opts.save_ckpt, 'checkpoint.pt')):
                os.remove(os.path.join(opts.save_ckpt, 'checkpoint.pt'))
            if os.path.exists(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt')):
                os.remove(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt'))
            os.rmdir(os.path.join(opts.save_ckpt))

    return {
            'Model' : opts.s_model, 'Dataset' : opts.dataset,
            'Policy' : opts.lr_policy, 'OS' : str(opts.output_stride), 'Epoch' : str(B_epoch),
            'F1 [0]' : "{:.2f}".format(B_val_score['Class F1'][0]), 'F1 [1]' : "{:.2f}".format(B_val_score['Class F1'][1])
            }