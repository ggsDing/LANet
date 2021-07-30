import datetime
import os

import numpy as np
import torch.autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import stats

import time
from datasets import RS_PD_random as RS
from skimage import io
import cv2
#################################
from models.LANet import LANet as Net
NET_NAME = 'LANet'
DATA_NAME = 'PD'

from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, intersectionAndUnion, AverageMeter, CaclTP

working_path = os.path.abspath('.')
args = {
    'gpu': True,
    's_class': 0,
    'val_batch_size': 1,
    'val_crop_size': 1024,
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'xxx.pth')
}

def norm_gray(x, out_range=(0, 255)):
    #x=x*(x>0)
    domain = np.min(x), np.max(x)
    #print(np.min(x))
    #print(np.max(x))
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0] + 1e-10)
    y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return y.astype('uint8')    

def main():
    net = Net(5, num_classes=RS.num_classes+1)    
    net.load_state_dict(torch.load(args['load_path']) )#, strict = False
    net = net.cuda()
    net.eval()
    print('Model loaded.')
    pred_path = os.path.join(RS.root, 'Eval', NET_NAME)
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')
        
    val_set = RS.RS('val', sliding_crop=True, crop_size=args['val_crop_size'], padding=False) # 
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    predict(net, val_loader, pred_path, args, f)
    f.close()

def predict(net, pred_loader, pred_path, args, f_out=None):
    acc_meter = AverageMeter()
    TP_meter = AverageMeter()
    pred_meter = AverageMeter()
    label_meter = AverageMeter()
    Union_meter = AverageMeter()
    output_info = f_out is not None

    for vi, data in enumerate(pred_loader):
        with torch.no_grad():
            img, label = data
            if args['gpu']:
                img = img.cuda().float()
                label = label.cuda().float()

            output, _, _ = net(img)
            
        output = output.detach().cpu()
        _, pred = torch.max(output, dim=1)
        pred = pred.squeeze(0).numpy()
        
        label = label.detach().cpu().numpy()
        acc, _ = accuracy(pred, label)
        acc_meter.update(acc)
        pred_color = RS.Index2Color(pred)
        img = img.detach().cpu().numpy().squeeze().transpose((1, 2, 0))[:,:,:3]
        img = norm_gray(img)
        pred_name = os.path.join(pred_path, '%d.png'%vi)
        io.imsave(pred_name, pred_color)
        TP, pred_hist, label_hist, union_hist = CaclTP(pred, label, RS.num_classes)
        TP_meter.update(TP)
        pred_meter.update(pred_hist)
        label_meter.update(label_hist)
        Union_meter.update(union_hist)
        print('Eval num %d/%d, Acc %.2f'%(vi, len(pred_loader), acc*100))    
        if output_info:
            f_out.write('Eval num %d/%d, Acc %.2f\n'%(vi, len(pred_loader), acc*100))   

    precision = TP_meter.sum / (label_meter.sum + 1e-10) + 1e-10
    recall = TP_meter.sum / (pred_meter.sum + 1e-10) + 1e-10
    F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    F1 = np.array(F1)
    IoU = TP_meter.sum / Union_meter.sum
    IoU = np.array(IoU)
    
    print(output.shape)
    print('Acc %.2f'%(acc_meter.avg*100))
    avg_F = F1[:-1].mean()
    mIoU = IoU[:-1].mean()
    print('Avg F1 %.2f'%(avg_F*100))
    print(np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    print('mIoU %.2f'%(mIoU*100))
    print(np.array2string(IoU * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    if output_info:
        f_out.write('Acc %.2f\n'%(acc_meter.avg*100))
        f_out.write('Avg F1 %.2f\n'%(avg_F*100))
        f_out.write(np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
        f_out.write('\nmIoU %.2f\n'%(mIoU*100))
        f_out.write(np.array2string(IoU * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    return avg_F


if __name__ == '__main__':
    main()
