import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import float32, optim
from torch.utils import data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np

import os
import random
import socket
from tqdm import tqdm

from PIL import Image

import utils
import network
import datasets as dt
from metrics import StreamSegMetrics
from utils import ext_transforms as et
from utils import histeq as hq

def get_dataset(opts):
    if opts.is_rgb:
        train_transform = et.ExtCompose([
            et.ExtResize(size=(496, 468)),
            et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor),
            et.ExtRandomVerticalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = et.ExtCompose([
            et.ExtResize(size=(496, 468)),
            et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = et.ExtCompose([
            et.ExtResize(size=(496, 468)),
            et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor),
            et.ExtRandomVerticalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485], std=[0.229])
        ])
        val_transform = et.ExtCompose([
            et.ExtResize(size=(496, 468)),
            et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485], std=[0.229])
        ])

    if opts.dataset == "CPN":
        train_dst = dt.CPNSegmentation(root=opts.data_root, datatype='CPN', image_set='train',
                                     transform=train_transform, is_rgb=True)
        val_dst = dt.CPNSegmentation(root=opts.data_root, datatype='CPN', image_set='val',
                                  transform=val_transform, is_rgb=True)
    elif opts.dataset == "CPN_all":
        train_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype='CPN_all', image_set='train',
                                     transform=train_transform, is_rgb=opts.is_rgb)
        val_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype='CPN_all', image_set='val',
                                  transform=val_transform, is_rgb=opts.is_rgb)
    else:
        train_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset, image_set='train',
                                     transform=train_transform, is_rgb=opts.is_rgb)
        val_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset, image_set='val',
                                  transform=val_transform, is_rgb=opts.is_rgb)
    
    return train_dst, val_dst


def save_val_image(opts, model, loader, device, epoch):

    if not os.path.exists(os.path.join(opts.save_val_dir, 'epoch_{}'.format(epoch))):
        try:
            os.mkdir(os.path.join(opts.save_val_dir, 'epoch_{}'.format(epoch)))
        except:
            raise Exception
    save_dir = os.path.join(os.path.join(opts.save_val_dir, 'epoch_{}'.format(epoch)))

    if opts.is_rgb:
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    else:
        denorm = utils.Denormalize(mean=[0.485], std=[0.229])

    for i, (images, labels) in tqdm(enumerate(loader)):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(images)
        probs = nn.Softmax(dim=1)(outputs)

        preds = torch.max(probs, 1)[1].detach().cpu().numpy()
        image = images.detach().cpu().numpy()
        lbl = labels.detach().cpu().numpy()
        #print('Image shape', image.shape) # (5, 1, 512, 512)
        #print('lablel shape', lbl.shape) # (5, 512, 512)
        
        for j in range(images.shape[0]):
            tar1 = (denorm(image[j]) * 255).transpose(1, 2, 0).astype(np.uint8)
            img = (denorm(image[j]) * 255).transpose(1, 2, 0)
            #print('denorm shape', tar1.shape) # (512, 512, 1)
            if not opts.is_rgb:
                tar1 = np.squeeze(tar1)
                img = np.squeeze(img).astype(np.float32)
            tar2 = (lbl[j] * 255).astype(np.uint8)
            tar3 = (preds[j] * 255).astype(np.uint8)

            tar4 = (255 - (lbl[j] * 160 + preds[j] * 95)).astype(np.uint8)
            #tar5 = ( (img + 0.2 * ( (255 - (preds[j] * 255).astype(np.float32)) )) / 1.2 ).astype(np.uint8)

            idx = str(i*images.shape[0] + j).zfill(3)
            Image.fromarray(tar4).save(os.path.join( save_dir, '{}_mask.png'.format(idx) ))
            if i*images.shape[0] + j < 5:
                Image.fromarray(tar1).save(os.path.join( save_dir, '{}_image.png'.format(idx) ))
            #Image.fromarray(tar5).save(os.path.join( save_dir, '{}_image.png'.format(idx) ))
            
            #Image.fromarray(tar3).save(os.path.join( save_dir, '{}_preds.png'.format(idx) ))
    

def validate(opts, model, loader, device, metrics, epoch, criterion):

    metrics.reset()
    ret_samples = []

    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = labels.detach().cpu().numpy()

            if opts.loss_type == 'ap_cross_entropy':
                weights = labels.detach().cpu().numpy().sum() / (labels.shape[0] * labels.shape[1] * labels.shape[2])
                weights = torch.tensor([weights, 1-weights], dtype=float32).to(device)
                criterion = utils.CrossEntropyLoss(weight=weights)
                loss = criterion(outputs, labels)
            elif opts.loss_type == 'ap_entropy_dice_loss':
                weights = labels.detach().cpu().numpy().sum() / (labels.shape[0] * labels.shape[1] * labels.shape[2])
                weights = torch.tensor([weights, 1-weights], dtype=float32).to(device)
                criterion = utils.EntropyDiceLoss(weight=weights)
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            metrics.update(target, preds)
            #loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

        if opts.save_val_results:
            save_val_image(opts, model, loader, device, epoch)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss  


def train(devices=None, opts=None, REPORT=None):

    if devices is None or opts is None:
        raise Exception
    
    LOGDIR = os.path.join(opts.Tlog_dir, opts.model, opts.current_time+'_'+opts.dataset)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    # Tensorboard 
    if opts.save_log:
        logdir = os.path.join(LOGDIR, 'log')
        writer = SummaryWriter(log_dir=logdir)
    # Validate
    if opts.save_val_results or opts.save_last_results:
        logdir = os.path.join(LOGDIR, 'val_results')
        os.mkdir(logdir)
        opts.save_val_dir = logdir
    # Train
    if opts.save_train_results:
        logdir = os.path.join(LOGDIR, 'train_results')
        os.mkdir(logdir)
        opts.save_train_dir = logdir
    # Check Point
    if opts.save_model:
        logdir = os.path.join(LOGDIR, 'best_param')
        os.mkdir(logdir)
        opts.save_ckpt = logdir
    else:
        logdir = os.path.join(LOGDIR, 'cache_param')
        os.mkdir(logdir)
        opts.save_ckpt = logdir
    # Save Options description
    with open(os.path.join(LOGDIR, 'summary.txt'), 'w') as f:
        for k, v in vars(opts).items():
            f.write("{} : {}\n".format(k, v))

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ''' (1) Get Datasets
    '''
    train_dst, val_dst = get_dataset(opts)
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size,
                                shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                shuffle=True, num_workers=2, drop_last=True)
    print("Dataset: %s, Train set: %d, Val set: %d" % 
                    (opts.dataset, len(train_dst), len(val_dst)))

    ''' (2) Load Model
    '''
    try:
        print("Model selection: {}".format(opts.model))
        if opts.model.startswith("deeplab"):
            model = network.model.__dict__[opts.model](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes, output_stride=opts.output_stride)
            if opts.separable_conv and 'plus' in opts.model:
                network.convert_to_separable_conv(model.classifier)
            utils.set_bn_momentum(model.backbone, momentum=0.01)
        else:
            model = network.model.__dict__[opts.model](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes)                            
    except:
        raise Exception
    
    ''' (2-1) Resume model
    '''
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(devices)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("Train from scratch...")
        resume_epoch = 0
        model.to(devices)

    ''' (3) Set up criterion
    '''
    if opts.loss_type == 'cross_entropy':
        criterion = utils.CrossEntropyLoss()
    elif opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(gamma=2, alpha=torch.tensor([0.02, 0.98]).to(devices))
    elif opts.loss_type == 'dice_loss':
        criterion = utils.DiceLoss()
    elif opts.loss_type == 'entropy_dice_loss':
        criterion = utils.EntropyDiceLoss()
    elif opts.loss_type == 'ap_cross_entropy':
        criterion = None
    elif opts.loss_type == 'ap_entropy_dice_loss':
        criterion = None
    else:
        raise NotImplementedError

    ''' (4) Set up optimizer
    '''
    if opts.model.startswith("deeplab"):
        optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), 
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

    ''' (5) Set up metrics
    '''
    metrics = StreamSegMetrics(opts.num_classes)
    early_stopping = utils.EarlyStopping(patience=opts.total_itrs * 0.1, verbose=True, 
                                            path=opts.save_ckpt, save_model=opts.save_model)
    dice_stopping = utils.DiceStopping(patience=opts.total_itrs * 0.1, verbose=True, 
                                            path=opts.save_ckpt, save_model=opts.save_model)
    best_score = 0.0

    ''' (.) Data Parallel
    '''
    # Data Parallel Option
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    ''' (.) Save best model
    '''
    def save_ckpt(path, cur_itrs):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)    

    ''' (6) Train
    '''
    B_epoch = 0
    B_val_score = None

    for epoch in range(resume_epoch, opts.total_itrs):

        model.train()
        running_loss = 0.0
        metrics.reset()

        for (images, lbl) in tqdm(train_loader, leave=True):

            images = images.to(devices)
            lbl = lbl.to(devices)
            
            optimizer.zero_grad()

            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()

            if opts.loss_type == 'ap_cross_entropy':
                weights = lbl.detach().cpu().numpy().sum() / (lbl.shape[0] * lbl.shape[1] * lbl.shape[2])
                weights = torch.tensor([weights, 1-weights], dtype=float32).to(devices)
                criterion = utils.CrossEntropyLoss(weight=weights)
                loss = criterion(outputs, lbl)
            elif opts.loss_type == 'ap_entropy_dice_loss':
                weights = lbl.detach().cpu().numpy().sum() / (lbl.shape[0] * lbl.shape[1] * lbl.shape[2])
                weights = torch.tensor([weights, 1-weights], dtype=float32).to(devices)
                criterion = utils.EntropyDiceLoss(weight=weights)
                loss = criterion(outputs, lbl)
            else:
                loss = criterion(outputs, lbl)

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
                model.eval()
                metrics.reset()
                val_score, val_loss = validate(opts, model, val_loader, 
                                                devices, metrics, epoch, criterion)

                print("[{}] Epoch: {}/{} Loss: {:.8f}".format('Validate', epoch+1, opts.total_itrs, val_loss))
                print(" Overall Acc: {:.2f}, Mean Acc: {:.2f}, FreqW Acc: {:.2f}, Mean IoU: {:.2f}".format(
                    val_score['Overall Acc'], val_score['Mean Acc'], val_score['FreqW Acc'], val_score['Mean IoU']))
                print(" Class IoU [0]: {:.2f} [1]: {:.2f}".format(val_score['Class IoU'][0], val_score['Class IoU'][1]))
                print(" F1 [0]: {:.2f} [1]: {:.2f}".format(val_score['Class F1'][0], val_score['Class F1'][1]))
                
                if early_stopping(val_loss, model):
                    B_epoch = epoch
                if dice_stopping(-1 * val_score['Class F1'][1], model):
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
            break

    tmp = []
    tmp.append("Model: {}, Datasets: {}".format(opts.model, opts.dataset))
    tmp.append("loss: {}, policy: {} lr: {}, os: {}".format(opts.loss_type, opts.lr_policy, 
                                                                opts.lr, opts.output_stride))
    tmp.append("Epoch [{}]".format(B_epoch))
    tmp.append("F1 \t\t [0]: {:.2f} [1]: {:.2f}".format(B_val_score['Class F1'][0], 
                                                            B_val_score['Class F1'][1]))
    tmp.append("Class IoU \t [0]: {:.2f} [1]: {:.2f}".format(B_val_score['Class IoU'][0], 
                                                                B_val_score['Class IoU'][1]))
    REPORT.append_msg(tmp)

    if opts.save_last_results:
        with open(os.path.join(LOGDIR, 'summary.txt'), 'a') as f:
            for k, v in B_val_score.items():
                f.write("{} : {}\n".format(k, v))

        if opts.save_model:
            model.load_state_dict(torch.load(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt')))
            save_val_image(opts, model, val_loader, devices, B_epoch)
            if os.path.exists(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt')):
                os.remove(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt'))
        else:
            model.load_state_dict(torch.load(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt')))
            save_val_image(opts, model, val_loader, devices, B_epoch)
            if os.path.exists(os.path.join(opts.save_ckpt, 'checkpoint.pt')):
                os.remove(os.path.join(opts.save_ckpt, 'checkpoint.pt'))
            if os.path.exists(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt')):
                os.remove(os.path.join(opts.save_ckpt, 'dicecheckpoint.pt'))
            os.rmdir(os.path.join(opts.save_ckpt))    