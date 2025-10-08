from datetime import datetime
import os
import os.path as osp
import timeit

import numpy as np
import pytz
import torch
import torch.nn.functional as F


import tqdm
import socket
from utils.metrics import *
from utils.Utils import *

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()

from utils.metric2 import mean_dice
from utils.metric2 import mean_assd
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

            predictions, _ = model(data)
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):

    def __init__(self, cuda, multiply_gpu, model, optimizer, scheduler, val_loader, domain_loader, out, max_epoch, stop_epoch=None,
                 lr=1e-3, interval_validate=10, interval_save=10, batch_size=8, warmup_epoch=10):
        self.cuda = cuda
        self.multiply_gpu = multiply_gpu
        self.warmup_epoch = warmup_epoch
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.lr = lr
        # self.lr_decrease_rate = lr_decrease_rate
        # self.lr_decrease_epoch = lr_decrease_epoch
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.domain_loader = domain_loader
        self.time_zone = 'Asia/Shanghai'
        self.timestamp_start = datetime.now(pytz.timezone(self.time_zone))

        self.interval_validate = interval_validate
        self.interval_save = interval_save

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)


        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_mean_dice = 0.0
        self.best_epoch = -1


    def validate_Whs(self):
        training = self.model.training
        res = eval(self.model, self.val_loader)


        self.model.eval()

        val_loss = 0.0
        data_num_cnt = 0.0
        metrics = []
        mean_dice = res
        is_best = mean_dice > self.best_mean_dice
        if is_best:
            self.best_epoch = self.epoch + 1
            self.best_mean_dice = mean_dice

            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.module.state_dict() if self.multiply_gpu else self.model.state_dict(),
                'learning_rate_gen': get_lr(self.optim),
                'best_mean_dice': self.best_mean_dice,
            }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
        else:
            if (self.epoch + 1) % self.interval_save == 0:
                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim.state_dict(),
                    'model_state_dict': self.model.module.state_dict() if self.multiply_gpu else self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


        if training:
            self.model.train()


    def train_epoch(self):
        self.model.train()
        self.running_seg_loss = 0.0

        start_time = timeit.default_timer()
        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.domain_loader), total=len(self.domain_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.domain_loader)
            self.iteration = iteration

            assert self.model.training

            self.optim.zero_grad()

            # train
            for param in self.model.parameters():
                param.requires_grad = True

            image = sample['image'].cuda()
            target_map = sample['label'].cuda()

            pred, _ = self.model(image)
            loss_seg = F.cross_entropy(pred, target_map)

            self.running_seg_loss += loss_seg.item()
            self.running_seg_loss /= len(self.domain_loader)

            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')

            loss_seg.backward()
            # loss_se.backward()
            self.optim.step()


        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Execution time: %.5f\n' %
              (self.epoch, get_lr(self.optim), self.running_seg_loss, stop_time - start_time))


    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            self.scheduler.step()

            if (self.epoch+1) % self.interval_validate == 0:
                self.validate_Whs()




