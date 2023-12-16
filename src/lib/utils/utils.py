from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import numpy as np
import os
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


def add_points(img_raw, pred_points, gt_points=None):

    h, w = img_raw.shape[-2:]
    pred_points[:, 0] *= h
    pred_points[:, 1] *= w

    # draw the predictions
    size = 2
    img_to_draw = np.array(img_raw.permute(1, 2 ,0).cpu()) # cv2.cvtColor(np.array(img_raw.permute(1, 2 ,0).cpu()), cv2.COLOR_RGB2BGR)
    img_to_draw = (img_to_draw * 255).astype(np.uint8).copy()

    if gt_points !=None:
        for p in gt_points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[1]), int(p[0])), size, (0, 255, 255), -1)

    for p in pred_points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[1]), int(p[0])), size, (0, 0, 255), -1)

    if gt_points !=None:
        cv2.putText(img_to_draw, f'GT: {len(gt_points)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)  # 在图片上写文字
    cv2.putText(img_to_draw, f'Pred: {len(pred_points)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 在图片上写文字

    # save the visualized image
    # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)
    return img_to_draw


def copy_cur_env(work_dir, dst_dir, exception):
    # if os.path.exists(dst_dir):
    #     shutil.rmtree(dst_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)