import argparse
import re
from torch import Tensor, einsum
from metric2 import mean_dice, mean_assd

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0')

# parser.add_argument('--model-file', type=str, default='logs_train/ct/ct-checkpoint_70.pth.tar')
parser.add_argument('--model-file', type=str, default='logs_train/mr/checkpoint_80.pth.tar') # source weight path

parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=20)
# parser.add_argument('--lr', type=float, default=5e-4)   #ct-mr
parser.add_argument('--lr', type=float, default=5e-6)   #mr-ct
parser.add_argument('--lr-decrease-rate', type=float, default=0.80, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='your data path')

parser.add_argument('--dataset', type=str, default='ct')
parser.add_argument('--model-source', type=str, default='mr')
# parser.add_argument('--dataset', type=str, default='mr')
# parser.add_argument('--model-source', type=str, default='ct')
parser.add_argument('--batch-size', type=int, default=16)

parser.add_argument('--model-ema-rate', type=float, default=0.98)
parser.add_argument('--pseudo-label-threshold', type=float, default=0.75)
parser.add_argument('--mean-loss-calc-bound-ratio', type=float, default=0.2)

args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import torch.nn.functional as F

import torch
torch.backends.cudnn.enabled = False
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from dataloaders import Whs_dataloader
from dataloaders import custom_transforms as trans
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import cv2
import torch.backends.cudnn as cudnn
import random
import glob
import sys

seed = 42
savefig = False
get_hd = True
model_save = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def dice_soft_loss(output, target):
    s = (10e-20)

    intersect = torch.sum(output * target)
    dice = (2 * intersect) / (torch.sum(output) + torch.sum(target) + s)

    return 1 - dice
def GenerateAffine(inputs, degreeFreedom=5, scale=[0.9, 1.2], shearingScale=[0.01, 0.01], Ngpu=0):
    degree = torch.FloatTensor(inputs.size(0)).uniform_(-degreeFreedom, degreeFreedom) * 3.1416 / 180;
    Theta_rotations = torch.zeros(inputs.size(0), 3, 3)

    Theta_rotations[:, 0, 0] = torch.cos(degree);
    Theta_rotations[:, 0, 1] = torch.sin(degree);
    Theta_rotations[:, 1, 0] = -torch.sin(degree);
    Theta_rotations[:, 1, 1] = torch.cos(degree);
    Theta_rotations[:, 2, 2] = 1

    degree = torch.FloatTensor(inputs.size(0), 2).uniform_(scale[0], scale[1])

    Theta_scale = torch.zeros(inputs.size(0), 3, 3)

    Theta_scale[:, 0, 0] = degree[:, 0]
    Theta_scale[:, 0, 1] = 0
    Theta_scale[:, 1, 0] = 0
    Theta_scale[:, 1, 1] = degree[:, 1]
    Theta_scale[:, 2, 2] = 1

    degree = torch.cat((torch.FloatTensor(inputs.size(0), 1).uniform_(-shearingScale[0], shearingScale[0]),
                        torch.FloatTensor(inputs.size(0), 1).uniform_(-shearingScale[1], shearingScale[1])), 1)

    Theta_shearing = torch.zeros(inputs.size(0), 3, 3)

    Theta_shearing[:, 0, 0] = 1
    Theta_shearing[:, 0, 1] = degree[:, 0]
    Theta_shearing[:, 1, 0] = degree[:, 1]
    Theta_shearing[:, 1, 1] = 1
    Theta_shearing[:, 2, 2] = 1

    Theta = torch.matmul(Theta_rotations, Theta_scale)
    Theta = torch.matmul(Theta_shearing, Theta)

    Theta_inv = torch.inverse(Theta)

    Theta = Theta[:, 0:2, :]
    Theta_inv = Theta_inv[:, 0:2, :]

    return Theta, Theta_inv

def apply_trasform(inputs, theta):
    grid = F.affine_grid(theta, inputs.size()).cuda()

    if len(inputs.size()) < 4:
        outputs = F.grid_sample(inputs, grid, mode='nearest', padding_mode="border")
    else:
        outputs = F.grid_sample(inputs, grid, padding_mode="border")

    return outputs

def norm_soft_size(a: Tensor, power:int) -> Tensor:
    b, c, w, h = a.shape
    sl_sz = w*h
    amax = a.max(dim=1, keepdim=True)[0]+1e-10
    resp = (torch.div(a,amax))**power
    ress = einsum("bcwh->bc", [resp]).type(torch.float32)
    ress_norm = ress/(torch.sum(ress,dim=1,keepdim=True)+1e-10)
    return ress_norm.unsqueeze(2)

def discrepancy_calconsis(v1, v2):
    assert v1.dim() == 4
    assert v2.dim() == 4
    n, c, h, w = v1.size()
    inner = torch.mul(v1, v2)
    v1 = v1.permute(2, 3, 1, 0)
    v2 = v2.permute(2, 3, 0, 1)
    mul = v1.matmul(v2)
    mul = mul.permute(2, 3, 0, 1)
    dis = torch.sum(mul) - torch.sum(inner)
    dis = dis / (h * w)
    return dis
def discrepancy_calc_contra(v1, v2):
    assert v1.dim() == 4
    assert v2.dim() == 4
    n, c, h, w = v1.size()
    inner = torch.mul(v1, v2)
    v1 = v1.permute(2, 3, 1, 0)
    v2 = v2.permute(2, 3, 0, 1)
    mul = v1.matmul(v2)
    mul = mul.permute(2, 3, 0, 1)

    return mul


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def soft_label_to_hard(soft_pls, pseudo_label_threshold):
    pseudo_labels = torch.zeros(soft_pls.size())
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[soft_pls > pseudo_label_threshold] = 1
    pseudo_labels[soft_pls <= pseudo_label_threshold] = 0

    return pseudo_labels



def adapt_epoch(model_t, model_s, optim, train_loader, args,
                feature_bank, pred_bank, loss_weight=None,):
    for sample_w, sample_s in train_loader:
        imgs_w = sample_w['image']
        imgs_s = sample_s['image']
        img_name = sample_w['img_name']
        if torch.cuda.is_available():
            imgs_w = imgs_w.cuda()
            imgs_s = imgs_s.cuda()

        # model predict
        predictions_stu_s1 = model_s(imgs_s)
        predictions_stu_s = predictions_stu_s1['out']
        with torch.no_grad():
            predictions_tea_w1 = model_t(imgs_w)
            predictions_tea_w = predictions_tea_w1['out']


        ###softmax
        predictions_stu_s_softmax: Tensor = F.softmax(predictions_stu_s, dim=1)
        predictions_tes_w_softmax: Tensor = F.softmax(predictions_tea_w, dim=1)

        pseudo_labels = soft_label_to_hard(predictions_tes_w_softmax, args.pseudo_label_threshold)

        loss_seg = F.cross_entropy(predictions_stu_s, pseudo_labels)

        f = lambda x: torch.exp(x / 0.07)
        z1z2 = discrepancy_calc_contra(f(predictions_stu_s_softmax), f(predictions_tes_w_softmax))
        z1z1 = discrepancy_calc_contra(f(predictions_stu_s_softmax), f(predictions_stu_s_softmax))

        diag_z1z2 = z1z2.diagonal(dim1=-2, dim2=-1).sum()
        diag_z1z1 = z1z1.diagonal(dim1=-2, dim2=-1).sum()
        # diag_z2z2 = z2z2.diagonal(dim1=-2, dim2=-1).sum()
        sum_z1z2 = z1z2.sum()
        sum_z1z1 = z1z1.sum()
        # sum_z2z2 = z2z2.sum()
        loss_contra = -torch.log(diag_z1z2 / sum_z1z2 + sum_z1z1 - diag_z1z1)

        ######################################################################################
        loss_consis = discrepancy_calconsis(predictions_stu_s_softmax,predictions_tes_w_softmax)
        ########################################################################
        loss = loss_contra + loss_seg + 1*loss_consis
        ####################################################cbmt

        Theta, Theta_inv = GenerateAffine(Variable(imgs_s, requires_grad=True))
        Theta2, Theta_inv2 = GenerateAffine(Variable(imgs_w, requires_grad=True))
        target_batch_images_aug = apply_trasform(Variable(imgs_s, requires_grad=True), Theta)
        target_batch_images_aug2 = apply_trasform(Variable(imgs_w, requires_grad=True), Theta2)
        #outputs_target_aug,features_outputs_target_aug = model_s(target_batch_images_aug)
        outputs_target_aug = model_s(target_batch_images_aug)
        outputs_target_aug = outputs_target_aug['out']
        with torch.no_grad():
            #outputs_target_aug2,features_outputs_target_aug2 = model_t(target_batch_images_aug2)
            outputs_target_aug2 = model_t(target_batch_images_aug2)
            outputs_target_aug2 = outputs_target_aug2['out']
        #outputs_target_transformed = apply_trasform(predictions_stu_s_sigmoid, Theta)
        outputs_target_transformed2 = apply_trasform(predictions_stu_s, Theta2)
        #consistency_loss = dice_soft_loss(torch.sigmoid(outputs_target_aug),
        #                                         torch.sigmoid(outputs_target_transformed))
        consistency_loss2 = dice_soft_loss(torch.sigmoid(outputs_target_aug2),
                                                 torch.sigmoid(outputs_target_transformed2))


        loss.backward()
        optim.step()
        optim.zero_grad()
        # update teacher
        for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
            param_t.data = param_t.data.clone() * args.model_ema_rate + param_s.data.clone() * (1.0 - args.model_ema_rate)


def eval(model, data_loader):
    model.eval()
    C = 5
    sample_dict = {}
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data = sample['image']
            target_map = sample['label']
            filenames_target = sample['img_name']
            data = data.cuda()

            predictions = model(data)
            predictions = predictions['out']
            for i, name in enumerate(filenames_target):
                sample_name, index = name.split("slice")[1].split("_")[0], int(name.split("slice")[1].split("_")[1].split('.')[0])
                sample_dict[sample_name] = sample_dict.get(sample_name, []) + [(predictions[i].detach().cpu(), target_map[i].detach().cpu(), index)]

        # PCA 3D
        pred_results_list = []
        gt_segs_list = []

        for k in sample_dict.keys():

            sample_dict[k].sort(key=lambda ele: ele[2])
            preds = []
            targets = []
            for pred, target, _ in sample_dict[k]:
                if target.sum() == 0:
                    continue
                preds.append(pred)
                targets.append(target)
            pred_results_list.append(torch.stack(preds, dim=-1))
            gt_segs_list.append(torch.stack(targets, dim=-1))

        res = mean_dice(pred_results_list, gt_segs_list, C)
        mean_assd(pred_results_list, gt_segs_list, C)

    model.train()

    return res


def main():
    now = datetime.now()
    here = osp.dirname(osp.abspath(__file__))
    args.out = osp.join(here, 'logs_target', args.dataset, now.strftime('%Y%m%d_%H%M%S.%f'))
    if not osp.exists(args.out):
        os.makedirs(args.out)
    args.out_file = open(osp.join(args.out, now.strftime('%Y%m%d_%H%M%S.%f')+'.txt'), 'w')
    args.out_file.write(' '.join(sys.argv) + '\n')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    # dataset
    composed_transforms_train = transforms.Compose([
        trans.Resize(256),
        trans.add_salt_pepper_noise(),
        trans.adjust_light(),
        trans.eraser(),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        trans.Resize(256),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])

    dataset_train = Whs_dataloader.WhsSegmentation_2transform(base_dir=args.data_dir, dataset=args.dataset,
                                                                    split='train',geshi='npy',
                                                                    transform_weak=composed_transforms_test,
                                                                    transform_strong=composed_transforms_train)
    dataset_train_weak = Whs_dataloader.WhsSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                              split='train',geshi='npy',
                                                              transform=composed_transforms_test)
    dataset_test = Whs_dataloader.WhsSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test',geshi='nii',
                                         transform=composed_transforms_test)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # train_loader_weak = DataLoader(dataset_train_weak, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model  2--5
    model_s = netd.DeepLab(num_classes=5, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)
    model_t = netd.DeepLab(num_classes=5, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)

    if torch.cuda.is_available():
        model_s = model_s.cuda()
        model_t = model_t.cuda()
    log_str = '==> Loading %s model file: %s' % (model_s.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model_s.state_dict()}
    model_s.load_state_dict(filtered_state_dict,strict=False)
    model_t.load_state_dict(filtered_state_dict,strict=False)


    if (args.gpu).find(',') != -1:
        model_s = torch.nn.DataParallel(model_s, device_ids=[0, 1])
        model_t = torch.nn.DataParallel(model_t, device_ids=[0, 1])

    optim = torch.optim.Adam(model_s.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model_s.train()
    model_t.train()
    for param in model_t.parameters():
        param.requires_grad = False

    # res = eval(model_t, test_loader)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    for epoch in range(args.epoch):

        log_str = '\nepoch {}/{}:'.format(epoch+1, args.epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        loss_weight = 0
        feature_bank = []
        pred_bank = []
        # adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank,loss_weight=loss_weight)
        adapt_epoch(model_t, model_s, optim, train_loader, args
                    ,pred_bank,feature_bank,loss_weight=loss_weight)

        scheduler.step()
        best_dice = 0
        print("teacher result:")
        res= eval(model_t, test_loader)
        if res > best_dice:
            torch.save({'model_state_dict': model_t.state_dict()}, args.out + '/'+ str(epoch+1) +'tbest_adaptation.pth.tar')
        print("student result:")
        res = eval(model_s, test_loader)
        if res > best_dice:
            torch.save({'model_state_dict': model_s.state_dict()}, args.out + '/' + str(epoch+1) + 'sbest_adaptation.pth.tar')


    torch.save({'model_state_dict': model_t.state_dict()}, args.out + '/after_adaptation.pth.tar')


if __name__ == '__main__':
    main()

