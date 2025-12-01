# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet3d.models.alocc.heads.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies, nusc_class_names
from mmdet3d.models.alocc.heads.occ_loss_utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from torch.utils.checkpoint import checkpoint as cp
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast
from mmdet3d.models import builder
from mmcv.cnn.bricks.conv_module import ConvModule
@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        soft_weights=False,
        loss_weight_cfg=None,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        final_occ_size=[256, 256, 20],
        empty_idx=0,
        balance_cls_weight=True,
        train_cfg=None,
        test_cfg=None,
        with_cp=False,
        use_focal_loss=False,
        use_dice_loss= False,
        use_deblock=True,
        CE_loss_only=False,
        process_feats_only=False,
        mid_channel=None,
        final_two_part_loss=False,
        pred_flow=False,
        flow_l2_loss=False,
        open_occ=False,
        flow_loss_weight=1.,
    ):
        super(OccHead, self).__init__()

        self.CE_loss_only=CE_loss_only
        self.process_feats_only=process_feats_only
        self.final_two_part_loss=final_two_part_loss
        self.pred_flow=pred_flow
        self.flow_l2_loss=flow_l2_loss
        
        self.fp16_enabled=False
      
        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.with_cp = with_cp
        
        self.use_deblock = use_deblock
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        


        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2 if mid_channel is None else mid_channel
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        if not self.process_feats_only:
            self.occ_pred_conv = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=mid_channel, 
                            out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, mid_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                            out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        self.soft_weights = soft_weights
        self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
            
        # loss functions
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.dice_loss = builder.build_loss(dict(type='DiceLoss', loss_weight=2))
        from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies
        
        if open_occ:
            nusc_class_frequencies=nusc_class_frequencies[np.array([4,10,9,3,5,2,6,7,8,1,11,12,13,14,15,16,17])]
        self.nusc_class_frequencies=nusc_class_frequencies

        if balance_cls_weight:
            if out_channel == 19 or out_channel == 18:
                self.class_weights = torch.from_numpy(1 / np.log(self.nusc_class_frequencies[:out_channel] + 0.001))
                self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])
            else:
                if out_channel == 17: self.nusc_class_frequencies[0] += self.nusc_class_frequencies[-1]
                self.class_weights = torch.from_numpy(1 / np.log(self.nusc_class_frequencies[:out_channel] + 0.001))
        else:
            self.class_weights = torch.ones(out_channel)/out_channel  # FIXME hardcode 17

        if self.use_deblock:
            upsample_cfg=dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[0],
                    out_channels=mid_channel,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, mid_channel)[1],
                                    nn.ReLU(inplace=True))


        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
        
        
        if self.pred_flow:
            self.flow_predicter = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=mid_channel, 
                            out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, mid_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                            out_channels=2, kernel_size=1, stride=1, padding=0))
            self.flow_loss_weight=flow_loss_weight
    
    @force_fp32(apply_to=('voxel_feats')) 
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}
        
        if self.use_deblock:
            if self.with_cp and voxel_feats[0].requires_grad:
                x0 = cp(self.deblock, voxel_feats[0])
            else:
                x0 = self.deblock(voxel_feats[0])
            output_occs.append(x0)
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp  and feats.requires_grad:
                x = cp(occ_conv, feats)
            else:
                x = occ_conv(feats)
            output_occs.append(x)

        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, H, W, D= output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        
        if self.process_feats_only:
            return [out_voxel_feats]
        
        output['out_voxel_feats'] = [out_voxel_feats]
        if self.with_cp and  out_voxel_feats.requires_grad:
            out_voxel = cp(self.occ_pred_conv, out_voxel_feats)
        else:
            out_voxel = self.occ_pred_conv(out_voxel_feats)

        output['occ'] = [out_voxel]
        
        if self.pred_flow:
            
            if self.with_cp and  out_voxel_feats.requires_grad:
                flow_pred = cp(self.flow_predicter, out_voxel_feats)
            else:
                flow_pred = self.flow_predicter(out_voxel_feats)
            output['flow'] = [flow_pred]
        return output
     
    @force_fp32()
    def forward(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, **kwargs):
        
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        output = self.forward_coarse_voxel(voxel_feats)
        if self.process_feats_only:
            return output
        out_voxel_feats = output['out_voxel_feats'][0]
        coarse_occ = output['occ'][0]

        res = {
            'output_voxels': output['occ'],
            'output_voxels_fine': output.get('fine_output', None),
            'output_coords_fine': output.get('fine_coord', None),
        }
        if 'flow' in output:
            res['output_flow'] = output['flow']


        return res
    
    @force_fp32()
    def forward_train(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, gt_occupancy=None, gt_occupancy_flow=None, **kwargs):
        res = self.forward(voxel_feats, img_feats=img_feats, pts_feats=pts_feats, transform=transform, **kwargs)
        if self.pred_flow:
            loss = self.loss(target_voxels=gt_occupancy,
                         output_voxels = res['output_voxels'],
                         output_coords_fine=res['output_coords_fine'],
                         output_voxels_fine=res['output_voxels_fine'],
                         gt_occ_flow=kwargs['gt_occ_flow'],
                         output_flow=res['output_flow'])
        else:
            loss = self.loss(target_voxels=gt_occupancy,
                output_voxels = res['output_voxels'],
                output_coords_fine=res['output_coords_fine'],
                output_voxels_fine=res['output_voxels_fine'])

        return loss,res['output_voxels']


    @force_fp32() 
    def loss_voxel(self, output_voxels, target_voxels, tag,output_flow=None,gt_occ_flow=None):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
        
        # output_voxels = torch.log(output_voxels * 0) + output_voxels/0 # debug !!!!!!!!

        output_voxels[torch.isnan(output_voxels)] = 0
        output_voxels[torch.isinf(output_voxels)] = 0
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        if self.CE_loss_only:
            
            ##########################
            preds=output_voxels.permute(0, 2, 3, 4, 1)
            mask=target_voxels!=255
            preds=preds[mask]
            target_voxels=target_voxels[mask]
            loss_dict['loss_voxel_ce_{}'.format(tag)] = torch.nn.CrossEntropyLoss()(preds, target_voxels)
            ##################
            # loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, None, ignore_index=255)
        else:
            if self.use_focal_loss:
                
                loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            else:
                loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

            loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
            loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)


        if self.use_dice_loss:
            visible_mask = target_voxels!=255
            visible_pred_voxels = output_voxels.permute(0, 2, 3, 4, 1)[visible_mask]
            visible_target_voxels = target_voxels[visible_mask]
            visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 19)
            loss_dict['loss_voxel_dice_{}'.format(tag)] = self.dice_loss(visible_pred_voxels, visible_target_voxels)
        if self.pred_flow:
            pred_flow=output_flow.permute(0, 2, 3, 4, 1)
            mask=gt_occ_flow!=float('inf')
            mask=mask[...,0]
            
            if mask.sum()==0:
                loss_dict['loss_flow'] = pred_flow[0][0][0][0].sum()*0.
            
            else:
                if self.flow_l2_loss:
                    loss_dict['loss_flow'] = F.mse_loss(pred_flow[mask],gt_occ_flow[mask])*self.flow_loss_weight
                else:
                    loss_dict['loss_flow'] = F.l1_loss(pred_flow[mask],gt_occ_flow[mask])*self.flow_loss_weight
  
        return loss_dict

    @force_fp32() 
    def loss(self, output_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None, visible_mask=None,gt_occ_flow=None,output_flow=None, **kwargs):
        loss_dict = {}
        if self.pred_flow:
            for index, output_voxel in enumerate(output_voxels):
                loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='c_{}'.format(index),output_flow=output_flow[index],gt_occ_flow=gt_occ_flow))
        else:
            for index, output_voxel in enumerate(output_voxels):
                loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='c_{}'.format(index)))
        return loss_dict
