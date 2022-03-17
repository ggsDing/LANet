import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms
import utils.transform as transform
from skimage.transform import rescale
from torchvision.transforms import functional as F

num_classes = 6
PD_COLORMAP = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 255],
                [0, 255, 0], [255, 255, 0], [255, 0, 0] ]
PD_CLASSES  = ['Invalid', 'Impervious surfaces','Building', 'Low vegetation',
                'Tree', 'Car', 'Clutter/background']
# PD_MEAN = np.array([0.33885107, 0.36215387, 0.33536868, 0.38485747])
# PD_STD  = np.array([0.14027526, 0.13798502, 0.14333207, 0.14513438])
PD_MEAN = np.array([85.8, 91.7, 84.9, 96.6, 47])
PD_STD  = np.array([35.8, 35.2, 36.5, 37, 55])

def BGRI2RGB(img):
    r = img[0, :, :]
    g = img[1, :, :]
    b = img[2, :, :]
    i = img[3, :, :]
    img = cv2.merge([r, g, b, i])
    return img

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im):
    return (im - PD_MEAN) / PD_STD

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(PD_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Index2Color(pred):
    colormap = np.asarray(PD_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def Colorls2Index(ColorLabels):
    for i, data in enumerate(ColorLabels):
        ColorLabels[i] = Color2Index(data)
    return ColorLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)

def get_file_name(mode='train'):
    assert mode in ['train', 'val']
    if mode == 'train':
        img_path = os.path.join(data_dir, 'train')
        pred_path = os.path.join(data_dir, 'numpy', 'train')
    else:
        img_path = os.path.join(data_dir, 'val')
        pred_path = os.path.join(data_dir, 'numpy', 'val')

    data_list = os.listdir(img_path)
    numpy_path_list = [os.path.join(pred_path, it) for it in data_list]
    return numpy_path_list

def read_RSimages(data_dir, mode):
    assert mode in ['train', 'val', 'test']
    if mode == 'test':
        img_path = os.path.join(data_dir, 'test')
        data_list = os.listdir(img_path)
        imgs = []
        for it in data_list:
            im = io.imread(os.path.join(img_path, it))
            imgs.append(im)
        return imgs

    img_path = os.path.join(data_dir, mode)
    dsm_path = os.path.join(data_dir, mode, 'dsm')
    mask_path = os.path.join(data_dir, 'groundtruth_noBoundary') #'groundtruth'
    data_list = os.listdir(img_path)
    data, labels = [], []
    count=0
    for it in data_list:
        # print(it)
        if (it[-4:]=='.tif'):
            dsm_name = 'dsm' + it[3:-10] + '.jpg'
            mask_name = it[:-10] + '_label_noBoundary.tif' #'_label.tif'
            fpath = os.path.join(img_path, it)
            dsm_fpath = os.path.join(dsm_path, dsm_name)
            mask_fpath = os.path.join(mask_path, mask_name)
            print(dsm_fpath)
            ext = os.path.splitext(it)[-1]
            if(ext == '.tif'):
                img = io.imread(fpath)
                dsm = io.imread(dsm_fpath)
                img = np.concatenate((img, np.expand_dims(dsm, axis=2)), axis=2)
                label = io.imread(mask_fpath)
                data.append(img)
                labels.append(label)
                count+=1
                #if count>1: break
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    return data, labels

def rescale_images(imgs, scale, order=0):
    for i, im in enumerate(imgs):
        imgs[i] = rescale_image(im, scale, order)
    return imgs
    
def rescale_image(img, scale=1/8, order=0):
    flag = cv2.INTER_NEAREST
    if order==1: flag = cv2.INTER_LINEAR
    elif order==2: flag = cv2.INTER_AREA
    elif order>2:  flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0]*scale), int(img.shape[1]*scale)),
                             interpolation=flag)
    return im_rescaled

class Loader(data.Dataset):
    def __init__(self, data_dir, mode, random_crop=False, crop_nums=40, random_flip = False, sliding_crop=False, crop_size=640/8, padding=False):
        self.crop_size = crop_size
        self.crop_nums = crop_nums
        self.random_flip = random_flip
        self.random_crop = random_crop
        data, labels = read_RSimages(data_dir, mode)
        if sliding_crop:
            data, labels = transform.create_crops(data, labels, [self.crop_size, self.crop_size])
        if padding:
            data, labels = transform.data_padding(data, labels, scale=16)
        self.data = data
        self.labels = Colorls2Index(labels)

        if self.random_crop:
            self.len = crop_nums*len(self.data)
        else:
            self.len = len(self.data)

    def __getitem__(self, idx):
        if self.random_crop:
            idx = int(idx/self.crop_nums)
            data, label = transform.random_crop(self.data[idx], self.labels[idx], size=[self.crop_size, self.crop_size])
        else:
            data = self.data[idx]
            label = self.labels[idx]
        if self.random_flip:
            data, label = transform.rand_flip(data, label)
            
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data, label

    def __len__(self):
        return self.len
