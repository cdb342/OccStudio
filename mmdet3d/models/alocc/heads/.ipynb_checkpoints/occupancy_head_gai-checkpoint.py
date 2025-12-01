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
from mmdet3d.models.fbbev.modules.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from mmdet3d.models.fbbev.modules.occ_loss_utils import nusc_class_frequencies, nusc_class_names
from mmdet3d.models.fbbev.modules.occ_loss_utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from torch.utils.checkpoint import checkpoint as cp
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast
from mmdet3d.models import builder
from mmcv.cnn.bricks.conv_module import ConvModule

@HEADS.register_module()
class OccHead_gai(BaseModule):
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
        #############
        CE_loss_only=False,
        use_predicter=True,
        out_dim=32,
        final_two_part_loss=False,
        binary_non_mask=False,
        flash_occ=False,
        Dz=16,
        num_cls=19,
        flash_occ_v2=False,
    ):
        super(OccHead_gai, self).__init__()
################

        self.CE_loss_only=CE_loss_only
        self.Dz=Dz
        self.num_cls=num_cls
        self.flash_occ_v2=flash_occ_v2
############
        self.fp16_enabled=False
      
        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.with_cp = with_cp
        # import pdb;pdb.set_trace()
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
        


        # # voxel-level prediction
        # self.occ_convs = nn.ModuleList()
        # for i in range(self.num_level):
        #     mid_channel = self.in_channels[i] // 2
        #     occ_conv = nn.Sequential(
        #         build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
        #                 out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
        #         build_norm_layer(norm_cfg, mid_channel)[1],
        #         nn.ReLU(inplace=True))
        #     self.occ_convs.append(occ_conv)


        # self.occ_pred_conv = nn.Sequential(
        #         build_conv_layer(conv_cfg, in_channels=mid_channel, 
        #                 out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
        #         build_norm_layer(norm_cfg, mid_channel//2)[1],
        #         nn.ReLU(inplace=True),
        #         build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
        #                 out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        # self.soft_weights = soft_weights
        # self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        # if self.soft_weights:
        #     soft_in_channel = mid_channel
        #     self.voxel_soft_weights = nn.Sequential(
        #         build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
        #                 out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
        #         build_norm_layer(norm_cfg, soft_in_channel//2)[1],
        #         nn.ReLU(inplace=True),
        #         build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
        #                 out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
            
        # loss functions
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.dice_loss = builder.build_loss(dict(type='DiceLoss', loss_weight=2))

        if balance_cls_weight:
            if out_channel == 19:
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:out_channel] + 0.001))
                self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])
            else:
                if out_channel == 17: nusc_class_frequencies[0] += nusc_class_frequencies[-1]
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:out_channel] + 0.001))
        else:
            self.class_weights = torch.ones(out_channel)/out_channel  # FIXME hardcode 17

        # if self.use_deblock:
        #     upsample_cfg=dict(type='deconv3d', bias=False)
        #     upsample_layer = build_conv_layer(
        #             upsample_cfg,
        #             in_channels=self.in_channels[0],
        #             out_channels=self.in_channels[0]//2,
        #             kernel_size=2,
        #             stride=2,
        #             padding=0)

        #     self.deblock = nn.Sequential(upsample_layer,
        #                             build_norm_layer(norm_cfg, self.in_channels[0]//2)[1],
        #                             nn.ReLU(inplace=True))


        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
        #############################
        self.flash_occ=flash_occ
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else out_channel
        if flash_occ or flash_occ_v2:
            conv_cfg=dict(type='Conv2d')
        else:
            conv_cfg=dict(type='Conv3d')
        self.final_conv = ConvModule(
                            self.in_channels[0],
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            conv_cfg=conv_cfg
                        )
        self.use_predicter =use_predicter
        if use_predicter:
            if flash_occ:
                 self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim * 2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim * 2, num_cls * Dz),
                )
            elif flash_occ_v2:
                self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim//Dz, self.out_dim//Dz*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim//Dz*2, out_channel),
                )
            else:
                self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim*2, out_channel),
                )
        self.final_two_part_loss=final_two_part_loss
        self.inter_binary_non_mask=binary_non_mask
        ##################################
    
    @force_fp32(apply_to=('voxel_feats')) 
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}
        # # import pdb;pdb.set_trace()
        # if self.use_deblock:
        #     if self.with_cp and voxel_feats[0].requires_grad:
        #         x0 = cp(self.deblock, voxel_feats[0])
        #     else:
        #         x0 = self.deblock(voxel_feats[0])
        #     output_occs.append(x0)
        # for feats, occ_conv in zip(voxel_feats, self.occ_convs):
        #     if self.with_cp  and feats.requires_grad:
        #         x = cp(occ_conv, feats)
        #     else:
        #         x = occ_conv(feats)
        #     output_occs.append(x)

        # if self.soft_weights:
        #     voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
        #     voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        # else:
        #     voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        # out_voxel_feats = 0
        # _, _, H, W, D= output_occs[0].shape
        # for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
        #     feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        #     out_voxel_feats += feats * weights.unsqueeze(1)
        # output['out_voxel_feats'] = [out_voxel_feats]
        # if self.with_cp and  out_voxel_feats.requires_grad:
        #     out_voxel = cp(self.occ_pred_conv, out_voxel_feats)
        # else:
        #     out_voxel = self.occ_pred_conv(out_voxel_feats)


        # import pdb;pdb.set_trace()
        out_voxel_feats=voxel_feats[0]

        if self.flash_occ:
            if self.with_cp and  out_voxel_feats.requires_grad:
                occ_pred =cp( self.final_conv,out_voxel_feats).permute(0, 3, 2, 1)
            else:
                occ_pred = self.final_conv(out_voxel_feats).permute(0, 3, 2, 1)
            output['out_voxel_feats'] = [occ_pred]   
            bs, Dx, Dy = occ_pred.shape[:3]
            if self.use_predicter:
                # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
                if self.with_cp and  occ_pred.requires_grad:
                    occ_pred = cp(self.predicter, occ_pred)
                else:
                    occ_pred = self.predicter(occ_pred)
                occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_cls)
                output['occ'] = [occ_pred.permute(0, 4,  2, 1,3)]

        elif self.flash_occ_v2:
            # import pdb;pdb.set_trace()
            if self.with_cp and  out_voxel_feats.requires_grad:
                occ_pred =cp( self.final_conv,out_voxel_feats).permute(0, 3, 2, 1)
            else:
                occ_pred = self.final_conv(out_voxel_feats).permute(0, 3, 2, 1)
            occ_pred=occ_pred.reshape(occ_pred.shape[0],occ_pred.shape[1],occ_pred.shape[2],self.Dz,occ_pred.shape[3]//self.Dz).permute(0,3,1,2,4)
            output['out_voxel_feats'] = [occ_pred]   
            # bs, Dx, Dy = occ_pred.shape[:3]
            if self.use_predicter:
                # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
                if self.with_cp and  occ_pred.requires_grad:
                    occ_pred = cp(self.predicter, occ_pred)
                else:
                    occ_pred = self.predicter(occ_pred)
                # occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_cls)
                output['occ'] = [occ_pred.permute(0, 4, 3, 2, 1)]


        else:
            if self.with_cp and  out_voxel_feats.requires_grad:
                occ_pred = cp(self.final_conv, voxel_feats[0]).permute(0, 4, 3, 2, 1)
            else:
                occ_pred = self.final_conv(voxel_feats[0]).permute(0, 4, 3, 2, 1)
            output['out_voxel_feats'] = [occ_pred]
            # bncdhw->bnwhdc
            if self.use_predicter:
                if self.with_cp and  occ_pred.requires_grad:
                    occ_pred = cp(self.predicter, occ_pred)
                else:
                    occ_pred = self.predicter(occ_pred)
            output['occ'] = [occ_pred.permute(0, 4, 3, 2, 1)]

        return output
     
    @force_fp32()
    def forward(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, **kwargs):
        
        # assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        output = self.forward_coarse_voxel(voxel_feats)
        out_voxel_feats = output['out_voxel_feats'][0]
        coarse_occ = output['occ'][0]

        res = {
            'output_voxels': output['occ'],
            'output_voxels_fine': output.get('fine_output', None),
            'output_coords_fine': output.get('fine_coord', None),
        }


        return res
    
    @force_fp32()
    def forward_train(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, gt_occupancy=None,gt_occupancy_ori=None, gt_occupancy_flow=None, **kwargs):
        res = self.forward(voxel_feats, img_feats=img_feats, pts_feats=pts_feats, transform=transform, **kwargs)
        loss = self.loss(target_voxels=gt_occupancy,
                         gt_occupancy_ori=gt_occupancy_ori,
            output_voxels = res['output_voxels'],
            output_coords_fine=res['output_coords_fine'],
            output_voxels_fine=res['output_voxels_fine'])

        return loss


    @force_fp32() 
    def loss_voxel(self, output_voxels, target_voxels,gt_occupancy_ori, tag):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        # import pdb;pdb.set_trace()
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
        if self.final_two_part_loss:
            loss_dict.update(self.TwoPart_loss(output_voxels.permute(0, 2, 3, 4, 1), gt_occupancy_ori,mask=target_voxels!=255))
        else:
            if self.CE_loss_only:
                # import pdb;pdb.set_trace()
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

        return loss_dict

    @force_fp32() 
    def loss(self, output_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None,gt_occupancy_ori=None, visible_mask=None, **kwargs):
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,gt_occupancy_ori,  tag='c_{}'.format(index)))
        return loss_dict
    @force_fp32() 
    def TwoPart_loss(self,inputs,targets,mask=None,occ_depth=None,occ_depth_gt=None):
        # import pdb;pdb.set_trace()
        loss = dict()
        # if self.add_occ_depth_loss:
        #     # import pdb;pdb.set_trace()
        #     depth_loss_occ=self.img_view_transformer.get_depth_loss_gt_occ2depth( occ_depth+1e-8,occ_depth_gt.permute(0,2,3,1).reshape(-1,occ_depth_gt.shape[1]))
        #     # depth_loss_occ=-((occ_depth+1e-8).log()*occ_depth_gt).sum(1).mean()
        #     loss['depth_loss_occ']=depth_loss_occ
        if mask is not None:
            mask=mask.bool()
            inputs_=inputs[mask]
            # import pdb;pdb.set_trace()
            targets_=targets[mask]
        else:
            inputs_=inputs.reshape(-1,inputs.shape[-1])
            targets_=targets.reshape(-1)
        sem=inputs_[:,:-1]
        occ=inputs_[:,-1]

        targets_occ=targets_!=18
        
        
        # mask_free=(targets==18).nonzero().squeeze()

        
        # if not self.only_sup_sem:
        if self.inter_binary_non_mask:
            targets=targets.reshape(-1)
            targets=targets!=18
            occ=inputs.reshape(-1,inputs.shape[-1])[:,-1]
            loss_occ=F.binary_cross_entropy_with_logits(occ,targets.float())
        else:
            loss_occ=F.binary_cross_entropy_with_logits(occ,targets_occ.float())
            # sem=(sem+1e-6).log()
            
        loss['loss_in_occ'] = loss_occ
        # if not self.sup_binary:
        
        
        mask_occ=(targets_occ).nonzero().squeeze()
        sem=sem[mask_occ]
        loss_sem=F.cross_entropy(sem,targets_[mask_occ])
        loss['loss_in_sem'] = loss_sem
        # loss=loss_occ+loss_sem
        # print((sem.sum(1)==0).sum(),2222222222222222222,loss_occ,loss_sem)
        return loss
