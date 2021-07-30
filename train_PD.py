import os
import time
import random
import numpy as np
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
working_path = os.path.dirname(os.path.abspath(__file__))

###############################################
from datasets import RS_PD_random as RS
from models.LANet import LANet as Net
NET_NAME = 'LANet'
DATA_NAME = 'PD'
###############################################

from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, intersectionAndUnion, AverageMeter

args = {
    'train_batch_size': 2,
    'val_batch_size': 2,
    'lr': 0.1,
    'epochs': 50,
    'gpu': True,
    'lr_decay_power': 1.5,
    'train_crop_size': 512,
    'crop_nums': 200,
    'val_crop_size': 512,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME, NET_NAME+'.png'),
    'chkpt_path': os.path.join(working_path, 'checkpoints', DATA_NAME, NET_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'SCAttNet_28e_OA90.29.pth')
}

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
writer = SummaryWriter(args['log_dir'])

def main():        
    net = Net(5, num_classes=RS.num_classes+1).cuda()
    #net.load_state_dict(torch.load(args['load_path']), strict=False)
    #for param in net.HR.parameters():
    #    param.requires_grad = False
    
    train_set = RS.RS('train', random_crop=True, crop_nums=args['crop_nums'], random_flip=True, crop_size=args['train_crop_size'], padding=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.RS('val', sliding_crop=True, crop_size=args['val_crop_size'])
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    train(train_loader, net, criterion, optimizer, scheduler, args, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, criterion, optimizer, scheduler, train_args, val_loader):
    bestaccT=0
    bestaccV=0.5
    bestloss=1
    begin_time = time.time()
    all_iters = float(len(train_loader)*args['epochs'])
    curr_epoch=0
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_main_loss = AverageMeter()
        
        curr_iter = curr_epoch*len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter+i+1
            adjust_lr_MP(optimizer, running_iter, all_iters)
            imgs, labels = data
            if args['gpu']:
                imgs = imgs.cuda().float()
                labels = labels.cuda().long()

            optimizer.zero_grad()
            outputs, aux1, aux2 = net(imgs)#

            assert outputs.shape[1] == RS.num_classes+1

            main_loss = criterion(outputs, labels)
            aux_loss1 = criterion(aux1, labels)
            aux_loss2 = criterion(aux2, labels)
            loss = main_loss + aux_loss1 *0.3 + aux_loss2 *0.3
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()
            _, preds = torch.max(outputs, dim=1)
            preds = preds.numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, valid_sum = accuracy(pred, label)
                # print(valid_sum)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_main_loss.update(loss.cpu().detach().numpy())

            # train_aux_loss.update(aux_loss, batch_pixel_sum)

            curr_time = time.time() - start

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_main_loss.val, acc_meter.val*100))
                writer.add_scalar('train loss', train_main_loss.val, running_iter)
                loss_rec = train_main_loss.val
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                # writer.add_scalar('train_aux_loss', train_aux_loss.avg, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)
                    
        acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch, train_args)
        if acc_meter.avg>bestaccT: bestaccT=acc_meter.avg
        if acc_v>bestaccV:
            bestaccV=acc_v
            bestloss=loss_v
            torch.save(net.state_dict(), args['chkpt_path']+'_%de_OA%.2f.pth'%(curr_epoch, acc_v*100))
        print('Total time: %.1fs Best rec: Train %.2f, Val %.2f, Val_loss %.4f' %(time.time()-begin_time, bestaccT*100, bestaccV*100, bestloss))
        curr_epoch += 1
        #scheduler.step()
        if curr_epoch >= train_args['epochs']:
            return

def validate(val_loader, net, criterion, curr_epoch, train_args):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs, labels = data

        if train_args['gpu']:
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()

        with torch.no_grad():
            outputs, _, _ = net(imgs)
            loss = criterion(outputs, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        _, preds = torch.max(outputs, dim=1)
        preds = preds.numpy()
        for (pred, label) in zip(preds, labels):
            acc, valid_sum = accuracy(pred, label)
            acc_meter.update(acc)

        if curr_epoch%args['predict_step']==0 and vi==0:
            pred_color = RS.Index2Color(preds[0])
            io.imsave(args['pred_dir'], pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Accuracy: %.2f'%(curr_time, val_loss.average(), acc_meter.average()*100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return acc_meter.avg, val_loss.avg

def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr

def adjust_lr_increase(optimizer, curr_iter, all_iter, init_lr=args['lr'], power=1.5):
    iter_rate = float(curr_iter) / all_iter
    
    running_lr = init_lr * iter_rate **power
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr

def adjust_lr_MP(optimizer, curr_iter, all_iter, init_lr=args['lr'], mid_lr=args['lr']/3, init_power=args['lr_decay_power'], mid_power=1.5):
    mid_iter = (1 - pow(mid_lr/init_lr, 1/init_power)) * all_iter
    if curr_iter<mid_iter:
        running_lr = init_lr * ((1. - float(curr_iter) / all_iter) ** init_power)
    else:
        running_lr = mid_lr * ((1. - float(curr_iter-mid_iter) / (all_iter-mid_iter)) ** mid_power)
                
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr
        
if __name__ == '__main__':
    main()
