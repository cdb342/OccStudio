import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Conv3d, caffe2_xavier_init
from mmcv.runner import ModuleList, force_fp32

from mmdet.models.builder import HEADS


from .base.anchor_free_head import AnchorFreeHead

import math
import os
from spatial_correlation_sampler import SpatialCorrelationSampler


def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res

@HEADS.register_module()
class ALOccFlowHead(AnchorFreeHead):
   
    def __init__(self,
                 feat_channels,
                 out_channels,
                pred_flow=False,
                flow_l2_loss=False,
                flow_loss_weight=1.0,
                do_history=True,
                history_cat_num=1,
                history_cat_conv_out_channels=None,
                single_bev_num_channels=80,
                interpolation_mode='bilinear',
                flow_with_his=False,
                use_adabin_flow_decoder=False,
                flow_scale=1.0,
                flow_cosine_loss=False,
                flow_out_channels=2,
                flow_bev=False,
                bev_out_channels=256,
                freeze_occ=False,
                bev_down_samp=False,
                adabin_fixed_bin=False,
                adabin_ce_sup=False,
                flow_his3=False,
                flow_his3_kernal_size=11,
                flow_his3_dilation=2,
                bev_3_9=False,
                cost_volum_temprature=0.1,

                num_cls=19,
                empty_idx=18,
     
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        #######
        self.use_adabin_flow_decoder=use_adabin_flow_decoder
        if self.use_adabin_flow_decoder:
            # flow_out_channels=88
            self.bin_invertal=[-22,22,0.5]
            flow_out_channels=len(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))*2
            self.flow_out_channels=flow_out_channels
            self.adabin_fixed_bin=adabin_fixed_bin
            self.flow_bin=torch.from_numpy(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))+0.5
            if adabin_fixed_bin:
                self.flow_bin=torch.from_numpy(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))
             
            else:
                if flow_bev:
                    self.adabin_bin_decoder = nn.Sequential(
                            nn.Linear(bev_out_channels, bev_out_channels),
                            nn.ReLU(inplace=True),
                            nn.Linear(bev_out_channels, flow_out_channels),

                        )
                else:
                    self.adabin_bin_decoder = nn.Sequential(
                                nn.Linear(feat_channels, feat_channels),
                                nn.ReLU(inplace=True),
                                nn.Linear(feat_channels, flow_out_channels),

                            )
            self.adabin_ce_sup=adabin_ce_sup
            
        self.semantic_cluster=semantic_cluster
        self.pred_flow=pred_flow
        self.flow_scale=flow_scale
        if self.pred_flow:
            
            if flow_his3:
                flow_post_conv_channels=int(feat_channels*1.5)
                flow_post_conv_mid_channels=feat_channels
            else:
                flow_post_conv_channels=feat_channels
                flow_post_conv_mid_channels=feat_channels
            self.flow_post_conv =  nn.Sequential(
                                nn.Conv3d(
                                    flow_post_conv_channels,
                                    flow_post_conv_mid_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1
                                ),
                    nn.ReLU(inplace=True),
                    )
            if flow_his3:
                flow_predicter_in_channels=feat_channels
            else:
                flow_predicter_in_channels=feat_channels
            if not flow_bev:
                self.flow_predicter = nn.Sequential(
                    nn.Conv3d(
                                    flow_predicter_in_channels,
                                    flow_predicter_in_channels*2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                ),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                                    flow_predicter_in_channels*2,
                                    flow_out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                )
                )
            else:
                self.flow_predicter = nn.Sequential(
                    nn.Conv2d(
                                    bev_out_channels,
                                    bev_out_channels*2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                                    bev_out_channels*2,
                                    flow_out_channels*16,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                )
                )
        self.flow_l2_loss=flow_l2_loss
        
        self.flow_loss_weight=flow_loss_weight
 
        
        self.flow_cosine_loss=flow_cosine_loss
        self.flow_his3=flow_his3
        if pred_flow:
            
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=21,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=2)
            
            cost_volumn_channel=feat_channels
            cost_volumn_channel=feat_channels//2
            if self.flow_his3:
                self.flow_his3_kernal_size=flow_his3_kernal_size
                self.flow_his3_dilation=flow_his3_dilation
                self.cost_volum_net=nn.Sequential(
                        nn.Conv2d(flow_his3_kernal_size*flow_his3_kernal_size, cost_volumn_channel, kernel_size=3,
                                    stride=1, padding=1),
                            nn.BatchNorm2d(cost_volumn_channel))
        self.flow_bev=flow_bev
        self.freeze_occ=freeze_occ
        self.bev_down_samp=bev_down_samp
        self.bev_3_9=bev_3_9

      
        ###################################

        self.flow_with_his=flow_with_his
        self.cost_volum_temprature=cost_volum_temprature
        if flow_with_his:
                    #######################################
                # Deal with history
            self.single_bev_num_channels = single_bev_num_channels
            self.do_history = do_history
            self.interpolation_mode = interpolation_mode
            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)

             ## Embed each sample with its relative temporal offset with current timestep
            conv =  nn.Conv3d
            
            self.history_sweep_time = None
            self.history_bev = None
            self.history_bev_before_encoder = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            self.count = 0
 

    @force_fp32()
    def fuse_history_corr(self, curr_bev, img_metas, bda,curr2=None,bev_2d=False,update_history=True): # align features with 3d shift
        # import pdb;pdb.set_trace()
        if bev_2d:
            curr_bev = curr_bev.unsqueeze(-1)
            if curr2 is not None:
                curr2 = curr2.unsqueeze(-1)
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
            if curr2 is not None:
                # import pdb;pdb.set_trace()
                curr2 = curr2.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        ## Deal with first batch

        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            if curr2 is not None:
                self.history_bev_gt=curr2.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
                if curr2 is not None:
                    self.history_bev_gt=curr2.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
                if curr2 is not None:
                    self.history_bev_gt[start_of_sequence]=curr2[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        if not bev_2d:
            # Generate grid
            xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
            ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
            zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
            grid = torch.stack(
                (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
            # import pdb;pdb.set_trace()
            # This converts BEV indices to meters
            # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
            # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
            feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)

            feat2bev[0, 0] = 0.4
            feat2bev[1, 1] =0.4
            feat2bev[2, 2] =0.4
            feat2bev[0, 3] =-40.
            feat2bev[1, 3] =-40.
            feat2bev[2, 3] = -1.
            # feat2bev[2, 2] = 1
            feat2bev[3, 3] = 1
            feat2bev = feat2bev.view(1,4,4)
            ## Get flow for grid sampling.
            # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
            # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
            # transform to previous grid locations.
            rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                    @ torch.inverse(forward_augs) @ feat2bev)

            grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
            tmp_bev = self.history_bev
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)
        else:
             # Generate grid
            xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
            ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
            grid = torch.stack(
                (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,1,4,1)

            # This converts BEV indices to meters
            # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
            # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
            feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
            feat2bev[0, 0] = 0.4
            feat2bev[1, 1] = 0.4
            feat2bev[0, 3] = -40.
            feat2bev[1, 3] = -40.
            feat2bev[2, 2] = 1
            feat2bev[3, 3] = 1
            feat2bev = feat2bev.view(1,4,4)
            ## Get flow for grid sampling.
            # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
            # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
            # transform to previous grid locations.
            rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                    @ torch.inverse(forward_augs) @ feat2bev)

            grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)

            grid = grid[:,:,:,:, :2,0] / normalize_factor.view(1, 1, 1, 1, 2) * 2.0 - 1.0

            tmp_bev = self.history_bev
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        
            sampled_history_bev = F.grid_sample(tmp_bev[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)

            if curr2 is not None and  self.training:
                sampled_history_bev_gt = F.grid_sample(self.history_bev_gt[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)
                
    
        feats_cat=sampled_history_bev
       
        feats_to_return=feats_cat
        if bev_2d:
            feats_to_return=feats_to_return.unsqueeze(2)
            if curr2 is not None and  self.training:
                sampled_history_bev_gt=sampled_history_bev_gt.unsqueeze(2)
        self.history_bev = curr_bev#[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_bev_gt=curr2
        
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
            if curr2 is not None and  self.training:
                sampled_history_bev_gt=sampled_history_bev_gt.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        if bev_2d:
            feats_to_return=feats_to_return.squeeze(-1)
            if curr2 is not None and  self.training:
                sampled_history_bev_gt=sampled_history_bev_gt.squeeze(-1)
        if curr2 is not None and  self.training:
            feats_to_return=[feats_to_return.clone()   ,sampled_history_bev_gt.clone()   ]
        else:
            feats_to_return=feats_to_return.clone()

        return feats_to_return

   
    def occ2bev(self,feat,occ_weight):
        
                    
        occ_weight=occ_weight.flip([-1])
        free_weight=1-occ_weight
        
        cum_free_weight=torch.cumprod(free_weight,dim=-1)
        cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1]),cum_free_weight[...,:-1]],dim=-1)
        weight=occ_weight*cum_free_weight
        weight=weight.flip([-1])
        out=(feat*weight.unsqueeze(1)).sum(-1)
        return out
    def flow_decoder(self, x,img_metas, occ_pred=None,num_cls=18,**kwargs):
        """Flow decoder.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map.
        """
        if self.flow_with_his:
  
            if self.flow_his3:
                with torch.no_grad():
                    occ_weight=(occ_pred!=num_cls-1).float()
                    
                        
                    if self.bev_3_9:
                        bev_feat_curr=x[...,3:9].mean(-1)
                    else:
                        bev_feat_curr=self.occ2bev(x,occ_weight)
                    
                    
                    bev_feat = self.fuse_history_corr(bev_feat_curr, img_metas, kwargs['bda'],bev_2d=True)
                    # import pdb;pdb.set_trace()
                    # bev_feat=bev_feat.mean(-1)
                    bev_feat_curr= F.interpolate(bev_feat_curr, [bev_feat_curr.shape[-2]//2,bev_feat_curr.shape[-1]//2], mode='bilinear', align_corners=True)
                    bev_feat= F.interpolate(bev_feat, [bev_feat.shape[-2]//2,bev_feat.shape[-1]//2], mode='bilinear', align_corners=True)
                    bev_feat_unfold=F.unfold(bev_feat,self.flow_his3_kernal_size,dilation=self.flow_his3_dilation,padding=self.flow_his3_kernal_size+self.flow_his3_dilation-3)
                    
                    bev_feat_unfold=bev_feat_unfold.reshape(*bev_feat.shape[:2],-1,*bev_feat.shape[-2:])
                    
                    
                    cost_volumn=-(F.normalize(bev_feat_curr,dim=1).unsqueeze(2)*F.normalize(bev_feat_unfold,dim=1)).sum(1)/self.cost_volum_temprature
                    invalid=bev_feat_unfold.sum(1)==0
                    # cost_volumn=cost_volumn.permute(0,2,3,1)
                    cost_volumn[invalid] = cost_volumn[invalid] + 5.
                        
                    cost_volumn = - cost_volumn
                    cost_volumn = cost_volumn.softmax(dim=1)
                    
                cost_volumn=self.cost_volum_net(cost_volumn)
                cost_volumn= F.interpolate(cost_volumn, [cost_volumn.shape[-2]*2,cost_volumn.shape[-1]*2], mode='bilinear', align_corners=True)
                cost_volumn=cost_volumn.unsqueeze(-1).repeat(1,1,1,1,x.shape[-1])
                bev_feat=torch.cat((cost_volumn,x),dim=1)
                # import pdb;pdb.set_trace()
                
                
            else:
                # import pdb;pdb.set_trace()
                bev_feat = self.fuse_history_corr(x.detach(), img_metas, kwargs['bda'])
                bev_feat=bev_feat.mean(-1)
                x=x.mean(-1)
                corr=self.correlation_sampler(x,bev_feat)
                corr=corr.reshape(corr.shape[0],-1,corr.shape[-2],corr.shape[-1])
                bev_feat=torch.cat((x,corr),dim=1)
            
   
        else:
            bev_feat = x

        
            
        flow_pred_ = self.flow_post_conv(bev_feat)
        if isinstance(flow_pred_, list):
            flow_pred_ = flow_pred_[0]
            # output['flow'] = [flow_pred]
        # import pdb;pdb.set_trace()
        flow_pred = self.flow_predicter(flow_pred_)
        if self.flow_bev:
            flow_pred = flow_pred.reshape(flow_pred.shape[0], -1,16, flow_pred.shape[2], flow_pred.shape[3])
            flow_pred=flow_pred.permute(0, 1, 3,4,2)

        if self.use_adabin_flow_decoder:
            # import pdb;pdb.set_trace()
            bin_weight=flow_pred[:,:,None,...]
            if self.adabin_fixed_bin:
                bin_center=self.flow_bin.to(flow_pred.device).float()
                bin_center=bin_center.unsqueeze(-1).unsqueeze(0).repeat(flow_pred.shape[0],1,2)
   
            else:    
                if self.flow_bev:
                    bin_prob=self.adabin_bin_decoder(flow_pred_.mean(dim=[2, 3]))
                else:
                    
                    bin_prob=self.adabin_bin_decoder(flow_pred_.mean(dim=[2, 3, 4]))
                # print(bin_prob.shape,1111111111111,self.flow_out_channels)
                # import pdb;pdb.set_trace()
                bin_prob=bin_prob.reshape(bin_prob.shape[0],self.flow_out_channels//2,2)
                bin_prob=bin_prob.softmax(1)
            

                cum_bin_prob=torch.cat((torch.zeros_like(bin_prob[:,:1,...]),torch.cumsum(bin_prob,1)[:,:-1]),dim=1)
                bin_center=self.bin_invertal[0]+(self.bin_invertal[1]-self.bin_invertal[0])*(bin_prob/2+cum_bin_prob)#BN,n_bin,H,W
                # import pdb;pdb.set_trace()
            bin_weight=bin_weight.reshape(bin_weight.shape[0],bin_weight.shape[1]//2,2,*bin_weight.shape[3:])
            # import pdb;pdb.set_trace()
            adabin_ce_loss=None
            
            if self.adabin_fixed_bin and self.adabin_ce_sup and self.training:
                # import pdb;pdb.set_trace()
                gt_flow=kwargs['gt_occ_flow']
                valid_mask=gt_flow!=float('inf')
                if valid_mask.sum()!=0:
                    gt_flow=gt_flow[valid_mask]
                    gt_flow=((gt_flow-self.bin_invertal[0])/self.bin_invertal[2]).round()
                    
                    # gt2=gt_flow.unsqueeze(1)-bin_center[0].unsqueeze(0)
                    # gt2=gt2.abs().argmin(1)
                    gt_flow=torch.clamp(gt_flow,0,bin_center.shape[1]-1)
                    gt_flow=gt_flow.long()
                    
                    bin_weight_valid=bin_weight.permute(0,3,4,5,2,1)[valid_mask]
                    adabin_ce_loss=F.cross_entropy(bin_weight_valid,gt_flow)
                else:
                    adabin_ce_loss=torch.zeros(1).to(gt_flow)
                
                
            bin_weight=bin_weight.softmax(1)
            flow_pred = torch.sum(bin_weight *bin_center[...,None,None,None], dim=1)
        # import pdb;pdb.set_trace()
        flow_pred=flow_pred.permute(0, 2, 3, 4, 1)
        # import pdb;pdb.set_trace()
        out=flow_pred*self.flow_scale
        if self.use_adabin_flow_decoder and self.adabin_fixed_bin and self.adabin_ce_sup:
            out=[out,adabin_ce_loss]
        
        return out
    def forward_train(self,
            feats,
            img_metas,
            gt_labels,
            mask_embede2=False,
            
            **kwargs,
        ):
        
        losses={}
        
        if self.pred_flow :

            flow_pred =self.flow_decoder(feats[0],img_metas, all_mask_preds[0].max(1)[1],num_cls=all_mask_preds[0].shape[1],**kwargs)
            if self.use_adabin_flow_decoder and self.adabin_fixed_bin and self.adabin_ce_sup:
                flow_pred,adabin_ce_loss=flow_pred
                losses.update({'adabin_ce_loss':adabin_ce_loss})
       
            gt_occ_flow=kwargs['gt_occ_flow']


            mask=gt_occ_flow!=float('inf')
            
            
            if  self.flow_l2_loss:
                self.flow_criterion=F.mse_loss
            else: 
                self.flow_criterion=F.l1_loss
            
            if mask.sum()==0:
                loss_flow=0.*flow_pred[0,0,0,0].sum()
            else:
                mask=mask[...,0]
                loss_flow= self.flow_criterion(flow_pred[mask],gt_occ_flow[mask])*self.flow_loss_weight
            losses.update({'loss_flow':loss_flow})
          

            if self.flow_cosine_loss:
                mask=gt_occ_flow!=float('inf')
                if mask.sum()==0:
                    loss_flow_cosine=0.*flow_pred[0,0,0,0].sum()
                else:

                    mask=mask[...,0]
                        
                    flow_pred_=flow_pred[mask]
                    gt_occ_flow_=gt_occ_flow[mask]
                    loss_flow_cosine=-(F.cosine_similarity(flow_pred_,gt_occ_flow_,dim=-1)).mean()*self.flow_loss_weight
                losses.update({'loss_flow_cosine':loss_flow_cosine})
            

    def simple_test(self, 
            feats,
            img_metas,
            **kwargs,
        ):
     
        if self.pred_flow:
            # import pdb;pdb.set_trace()
           
            flow_pred =self.flow_decoder(feats[0],img_metas, output_voxels.max(1)[1],num_cls=all_mask_preds[0].shape[1],**kwargs)
            if self.use_adabin_flow_decoder and self.adabin_fixed_bin and self.adabin_ce_sup:
                flow_pred,_=flow_pred
            # import pdb;pdb.set_trace()
            # flow_pred=flow_pred.permute(0, 2, 3, 4, 1)
            # gt_occ_flow=kwargs['gt_occ_flow']
            # mask=gt_occ_flow!=float('inf')
        
            # loss_flow= F.l1_loss(flow_pred[mask],gt_occ_flow[mask])
            # losses.update({'loss_flow':loss_flow})
            res['output_flow'] = [flow_pred]

        return res
   