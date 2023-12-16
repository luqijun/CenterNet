from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np

from ..models.losses import FocalLoss
from ..models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from ..models.decode import ctdet_decode
from ..models.utils import _sigmoid
from ..utils.debugger import Debugger
from ..utils.post_process import ctdet_post_process
from ..utils.oracle_utils import gen_oracle_map
from ..utils.matcher import HungarianMatcher
from .base_trainer import BaseTrainer


class CtCrowdLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtCrowdLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.matcher = HungarianMatcher(1, 0.05)

        empty_weight = torch.Tensor([0.5, 1.0])
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_grid_points(self, image_shape, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        opt = self.opt
        stride = opt.down_ratio

        # get image shape
        shape = (image_shape + stride // 2 - 1) // stride

        # generate point queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        grid_points = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1, 0)  # 2xN --> Nx2

        return grid_points.float()

    def forward(self, outputs, batch):
        opt = self.opt

        #region 方法一 生成grid points，计算分类损失和点位置损失
        outputs = outputs[0]
        bs, c, h, w = outputs['wh'].shape
        img_shape = torch.tensor(outputs['wh'].shape[2:]) * opt.down_ratio
        img_h, img_w = img_shape

        grid_points = self.get_grid_points(img_shape).unsqueeze(0).to(device=opt.device)
        grid_points[..., 0] /= img_h
        grid_points[..., 1] /= img_w

        pred_logits =  outputs['wh'].permute(0, 2, 3, 1).view(bs, -1, 2)
        outputs['reg']= (outputs['reg'].sigmoid() - 0.5) * 4.0
        offsets = outputs['reg'].permute(0, 2, 3, 1).view(bs, -1, 2)
        offsets[..., 0] /= img_h
        offsets[..., 1] /= img_w

        # 加了效果不好？
        # if img_shape[0]>256:
        #     offsets[...,0] /= (img_shape[0] / 256)
        # if img_shape[1]>256:
        #     offsets[...,1] /= (img_shape[1] / 256)

        pred_points = grid_points + offsets

        # if pred_points.max()>2:
        #     return

        # match
        outputs['img_shape'] = img_shape
        outputs["pred_logits"] = pred_logits
        outputs["pred_points"] = pred_points
        targets = batch['targets']
        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)

        # 计算分类损失
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        # 计算点位置损失
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        # loss_points = F.smooth_l1_loss(src_points, target_points)
        num_points = sum(len(t["labels"]) for t in targets)
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')
        loss_points = loss_points_raw.sum() / num_points

        weight_dict = [1.0, 0.5]

        loss = loss_ce * weight_dict[0] + loss_points * weight_dict[1]

        # 筛选点
        thrs = 0.5
        out_scores = torch.nn.functional.softmax(pred_logits, -1)[..., 1]
        valid = out_scores > thrs
        val_idx = valid.cpu()
        final_points = pred_points[val_idx]

        # process predicted points
        predict_cnt = torch.Tensor([sum(single_map).item() for single_map in val_idx])
        gt_cnt = torch.Tensor([len(t["labels"]) for t in targets])  #  len(target_classes_o) # targets[0]['points'].shape[0]
        mae = abs(predict_cnt - gt_cnt).mean()
        mse = ((predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)).mean()


        loss_stats = {'loss': loss, 'hm_loss': torch.Tensor([0.0]),
                      'wh_loss': loss_ce , 'off_loss': torch.Tensor([0.0]),
                      'point_loss': loss_points, 'mae': mae, 'mse': mse,
                      'pred_points': final_points }

        return loss, loss_stats

        #endregion


    def forward_old(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class CtCrowdTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtCrowdTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'point_loss', 'mae', 'mse']
        loss = CtCrowdLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]



