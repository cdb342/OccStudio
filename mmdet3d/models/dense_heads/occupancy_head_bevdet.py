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
import math
@HEADS.register_module()
class OccHead_BEVDet(BaseModule):
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
        use_predicter=True,
        out_dim=32,
        final_two_part_loss=False,
        binary_non_mask=False,
        flash_occ=False,
        Dz=16,
        num_cls=19,
        flash_occ_v2=False,
        pred_flow=False,
        wo_pred_occ=False,
        flow_l2_loss=False,
        open_occ=False,
        flow_loss_weight=1.,
    ):
        super(OccHead_BEVDet, self).__init__()
        self.CE_loss_only=CE_loss_only
        self.Dz=Dz
        self.num_cls=num_cls
        self.flash_occ_v2=flash_occ_v2
        self.pred_flow=pred_flow
        self.wo_pred_occ=wo_pred_occ
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

        # loss functions
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.dice_loss = builder.build_loss(dict(type='DiceLoss', loss_weight=2))
        from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies
        self.nusc_class_frequencies=nusc_class_frequencies
        if open_occ:
            nusc_class_frequencies=nusc_class_frequencies[np.array([4,10,9,3,5,2,6,7,8,1,11,12,13,14,15,16,17])]
        if balance_cls_weight:
            if out_channel == 19 or out_channel == 18:
                self.class_weights = torch.from_numpy(1 / np.log(self.nusc_class_frequencies[:out_channel] + 0.001))
                self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])
            else:
                if out_channel == 17: self.nusc_class_frequencies[0] += self.nusc_class_frequencies[-1]
                self.class_weights = torch.from_numpy(1 / np.log(self.nusc_class_frequencies[:out_channel] + 0.001))
        else:
            self.class_weights = torch.ones(out_channel)/out_channel  # FIXME hardcode 17

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
        if not self.wo_pred_occ:
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
        if self.pred_flow:
            self.flow_predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.ReLU(),
                nn.Linear(self.out_dim*2, 2),
            )
            self.final_conv_flow = ConvModule(
                            self.in_channels[0],
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            conv_cfg=conv_cfg
                        )
            self.flow_loss_weight=flow_loss_weight

    @force_fp32(apply_to=('voxel_feats')) 
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}

        out_voxel_feats=voxel_feats[0]
        if not self.wo_pred_occ:
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
                
                
                
        else:
            output['out_voxel_feats'] =[None]
            output['occ'] = [None]

        if self.pred_flow:
            if self.with_cp and  out_voxel_feats.requires_grad:
                flow_pred = cp(self.final_conv_flow, voxel_feats[0]).permute(0, 4, 3, 2, 1)
            else:
                flow_pred = self.final_conv_flow(voxel_feats[0]).permute(0, 4, 3, 2, 1)
            # output['flow'] = [flow_pred]
            if self.with_cp and  flow_pred.requires_grad:
                flow_pred = cp(self.flow_predicter, flow_pred)
            else:
                flow_pred = self.flow_predicter(flow_pred)
            output['flow'] = [flow_pred.permute(0, 4, 3, 2, 1)]

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
        if 'flow' in output:
            res['output_flow'] = output['flow']

        if 'occ_weight' in output:
            res['occ_weight']=output['occ_weight']

        return res
    
    @force_fp32()
    def forward_train(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, gt_occupancy=None,gt_occupancy_ori=None, gt_occupancy_flow=None, **kwargs):
        res = self.forward(voxel_feats, img_feats=img_feats, pts_feats=pts_feats, transform=transform, **kwargs)
        if self.pred_flow:
            loss = self.loss(target_voxels=gt_occupancy,
                         gt_occupancy_ori=gt_occupancy_ori,
                         output_voxels = res['output_voxels'],
                         output_coords_fine=res['output_coords_fine'],
                         output_voxels_fine=res['output_voxels_fine'],
                         gt_occ_flow=kwargs['gt_occ_flow'],
                         output_flow=res['output_flow'],res=res, **kwargs)
        else:
            loss = self.loss(target_voxels=gt_occupancy,
                            gt_occupancy_ori=gt_occupancy_ori,
                output_voxels = res['output_voxels'],
                output_coords_fine=res['output_coords_fine'],
                output_voxels_fine=res['output_voxels_fine'],res=res, **kwargs)

        return loss,res['output_voxels']


    @force_fp32() 
    def loss_voxel(self, output_voxels, target_voxels,gt_occupancy_ori, tag,output_flow=None,gt_occ_flow=None,sup_binary_only=False, res=None,**kwargs):
        
        # resize gt     
        if sup_binary_only:
            non_empty_mask=(target_voxels!=self.num_cls-1)*(target_voxels!=255)==1
            target_voxels[non_empty_mask]=0
        if not self.wo_pred_occ:                  
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

            output_voxels[torch.isnan(output_voxels)] = 0
            output_voxels[torch.isinf(output_voxels)] = 0
            assert torch.isnan(output_voxels).sum().item() == 0
            assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        if self.final_two_part_loss:
            loss_dict.update(self.TwoPart_loss(output_voxels.permute(0, 2, 3, 4, 1), gt_occupancy_ori,mask=target_voxels!=255))
        else:
            if not self.wo_pred_occ:
                if self.CE_loss_only:
                    
                    ##########################
                    preds=output_voxels.permute(0, 2, 3, 4, 1)
                    mask=target_voxels!=255
                    preds=preds[mask]
                    target_voxels=target_voxels[mask]
                    loss_dict['loss_voxel_ce_{}'.format(tag)] = torch.nn.CrossEntropyLoss()(preds, target_voxels.long())
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
                target_voxels=None,gt_occupancy_ori=None, visible_mask=None,gt_occ_flow=None,output_flow=None,res=None, **kwargs):
       
        loss_dict = {}
        if self.pred_flow:
            for index, output_voxel in enumerate(output_voxels):
                loss_dict.update(self.loss_voxel(output_voxel, target_voxels,gt_occupancy_ori,  tag='c_{}'.format(index),output_flow=output_flow[index],gt_occ_flow=gt_occ_flow,res=res, **kwargs))
        else:
            for index, output_voxel in enumerate(output_voxels):
                loss_dict.update(self.loss_voxel(output_voxel, target_voxels,gt_occupancy_ori,  tag='c_{}'.format(index),res=res, **kwargs))
        return loss_dict
    @force_fp32() 
    def TwoPart_loss(self,inputs,targets,mask=None,occ_depth=None,occ_depth_gt=None):
        
        loss = dict()

        if mask is not None:
            mask=mask.bool()
            inputs_=inputs[mask]
            
            targets_=targets[mask]
        else:
            inputs_=inputs.reshape(-1,inputs.shape[-1])
            targets_=targets.reshape(-1)
        sem=inputs_[:,:-1]
        occ=inputs_[:,-1]

        targets_occ=targets_!=18

        
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

        return loss
