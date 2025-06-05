#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   rotated_vild.py
@Version      :   1.0
@Time         :   2024/09/07 21:28:58
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   Implementation of Rotated ViLD.

Ref: https://arxiv.org/abs/2104.13921
'''

import torch
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models import TwoStageDetector
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
import torch.nn as nn
from torchvision.transforms import InterpolationMode, Resize, Compose, CenterCrop
import numpy as np
import torch.nn.functional as F
from mmcv.ops import nms_rotated
from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
import copy
import os
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes

@MODELS.register_module()
class RotatedViLD(TwoStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 rpn_bbox_type: str='xywha',
                 visual: ConfigType=None,
                 pseudo_cfg: ConfigType=None,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        
        self.proposal_path = pseudo_cfg.get('proposal_path', None)
        vector_path = pseudo_cfg.get('vector_path', None)
        self.words = nn.Parameter(torch.tensor(np.load(vector_path)), requires_grad=False) if vector_path is not None else None
        self.pseudo_nms = pseudo_cfg.get('pseudo_nms', True)
        self.iou_threshold = pseudo_cfg.get('iou_threshold', 0.6)
        self.pre_keep = pseudo_cfg.get('pre_keep', 64)
        self.post_keep = pseudo_cfg.get('post_keep', 16)
        self.semi_weight = pseudo_cfg.get('semi_weight', 0.5)
        self.train_with_gt = pseudo_cfg.get('train_with_gt', True)
        self.initialize = pseudo_cfg.get('initialize', True)
        self.clip_logit_scale = pseudo_cfg.get('clip_logit_scale', 100.0)
        self.mini_batch_size = pseudo_cfg.get('mini_batch_size', 128)
        self.filter_empty_instances = pseudo_cfg.get('filter_empty_instances', False)
        self.rpn_bbox_type = rpn_bbox_type

        self.visual = MODELS.build(visual) if visual is not None else None
        self.resize = Compose([
                Resize(size=visual.image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=(visual.image_size, visual.image_size)),
            ])


    def init_pseudos(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        if self.initialize:
            x = self.extract_feat(batch_inputs)
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
            
            rois = bbox2roi([res.bboxes for res in rpn_results_list])
            bbox_feats = self.roi_head.bbox_roi_extractor(
                x[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
            if self.roi_head.with_shared_head:
                bbox_feats = self.roi_head.shared_head(bbox_feats)
            cls_score, reg_bboxes = self.roi_head.bbox_head(bbox_feats)

            bbox_pred = self.roi_head.bbox_head.bbox_coder.decode(rois[:, 1:], reg_bboxes)
            num_per_img = [len(r) for r in rpn_results_list]
            bbox_pred = bbox_pred.split(num_per_img)
            
            for idx, (input, data_samples, rpn_results) in enumerate(zip(batch_inputs, batch_data_samples, rpn_results_list)):
                rpn_results.bboxes = bbox_pred[idx]
                _, ids = nms_rotated(rpn_results.bboxes.tensor, rpn_results.scores, self.iou_threshold)
                ids = ids[:self.pre_keep]
                rpn_results = rpn_results[ids]

                crop_batch = self.crop_images(input.unsqueeze(0), [rpn_results])
                clip_feature = torch.cat([
                    F.normalize(self.get_clip_features(mini_batch), dim=-1) for mini_batch in crop_batch.split(self.mini_batch_size)
                ], dim=0)
                cls_score, labels = (self.clip_logit_scale * clip_feature @ self.words.T).softmax(dim=-1)[:,:-1].max(dim=-1)

                _, ids = nms_rotated(rpn_results.bboxes.tensor, cls_score, self.iou_threshold)
                ids = ids[:self.post_keep]
                bboxes = rpn_results.bboxes[ids]
                bboxes.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.device))
                self.save_one_pseudo(os.path.basename(data_samples.img_path),
                                    clip_feature[ids], bboxes.tensor)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        self.init_pseudos(batch_inputs, batch_data_samples, rescale)
        predictions = super().predict(batch_inputs, batch_data_samples, rescale)
        return predictions

    def save_one_pseudo(self, img_name, features, bboxes):
        save_path = os.path.join(self.proposal_path, img_name.split('.')[0]+'.npz')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pseudo_labels = {
            'features': features.detach().cpu().numpy(),
            'bboxes': bboxes.detach().cpu().numpy(),
        }
        np.savez(save_path, **pseudo_labels)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        losses = self.sup_loss(batch_inputs, batch_data_samples)
        losses.update(**self.loss_by_unsup_instances(batch_inputs, batch_data_samples))

        return losses

    def sup_loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:

        if self.filter_empty_instances:
            keep_ids = [idx for idx, data_sample in enumerate(batch_data_samples) if len(data_sample.gt_instances) > 0]
            batch_inputs = batch_inputs[keep_ids]
            batch_data_samples = [batch_data_samples[i] for i in keep_ids]
            if len(batch_data_samples) == 0:
                return dict()

        losses = super().loss(batch_inputs, batch_data_samples)

        return losses

    @property
    def device(self) -> torch.device:
        return self.data_preprocessor.device


    def get_distill_features(self, batch_data_samples: SampleList):
        features, bboxes = [], []
        for data_samples in batch_data_samples:
            img_name = os.path.basename(data_samples.img_path)
            proposal_path = os.path.join(self.proposal_path, img_name.split('.')[0]+'.npz')
            data = np.load(proposal_path, allow_pickle=True)
            features_, bboxes_ = dict(data).values()
            features_ = torch.tensor(features_, device=self.device).squeeze(0)
            bboxes_ = RotatedBoxes(torch.tensor(bboxes_, device=self.device).squeeze(0))
            bboxes_.project_(torch.from_numpy(data_samples.homography_matrix).to(self.device))
            
            # convert to rpn bbox type
            if self.rpn_bbox_type == 'xywh':
                bboxes_ = bboxes_.convert_to('hbox')

            features.append(features_)
            bboxes.append(bboxes_)
        
        rois = bbox2roi(bboxes)
        clip_feature = torch.cat(features, dim=0)
        return rois, clip_feature

    def loss_by_unsup_instances(self, batch_inputs: Tensor,
                            batch_data_samples: SampleList) -> dict:
        
        x = self.extract_feat(batch_inputs)

        rois, clip_feature = self.get_distill_features(batch_data_samples)

        # assert self.with_bbox, 'Bbox head must be implemented.'
        # x = self.extract_feat(batch_inputs)

        bbox_feats = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
        cls_feature = self.roi_head.bbox_head.forward_cls_feature(bbox_feats, only_feature=True)

        losses = dict()
        if cls_feature.numel() > 0:
            losses['loss_dist'] = F.l1_loss(
                F.normalize(clip_feature, dim=-1),
                F.normalize(cls_feature, dim=-1)
            )
            losses = rename_loss_dict('semi_', reweight_loss_dict(losses, clip_feature.shape[-1] * self.semi_weight))

        return losses

    
    def crop_images(self, batch_inputs: Tensor,
                    rpn_results_list: InstanceList,) -> Tensor:
        batch_crops = []
        for batch, results in zip(batch_inputs, rpn_results_list):
            image_size = (batch.shape[-1], batch.shape[-2])
            for bbox in results.bboxes:
                coor = self.get_coordinate(bbox, 0.0, image_size, min_size=(200,200))
                batch_crops.append(self.resize(batch[:,coor[1]:coor[3],coor[0]:coor[2]])[None,:])
        batch_crops = torch.cat(batch_crops, dim=0)
        return batch_crops

    @torch.no_grad()
    def get_clip_features(self, batch_inputs: Tensor) -> Tensor:
        self.visual.eval()
        return self.visual(batch_inputs)

    def get_coordinate(self, gt_bbox, ratio=0, image_size=(800, 800), min_size=(-1, -1), square=False):
        # image_size(w, h)
        if isinstance(gt_bbox, RotatedBoxes):
            x0, y0, x1, y1 = gt_bbox.convert_to('hbox').tensor.squeeze().tolist()
        else:
            raise NotImplementedError
            x0, y0, x1, y1 = gt_bbox

        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + w//2, y0 + h//2
        
        w = min(image_size[0], max(w*(1+ratio), min_size[0]))
        h = min(image_size[1], max(h*(1+ratio), min_size[1]))
        if square:
            w = h = min(image_size[0], image_size[1], max(w, h))
        
        cx = w//2 if cx < w//2 else cx
        cx = image_size[0]-w//2 if cx > image_size[0]-w//2 else cx
        cy = h//2 if cy < h//2 else cy
        cy = image_size[1]-h//2 if cy > image_size[1]-h//2 else cy
        x0, x1, y0, y1 = int(cx-w//2), int(cx+w//2), int(cy-h//2), int(cy+h//2)

        return (x0, y0, x1, y1)