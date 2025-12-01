import torch
import torch.nn.functional as F
import torch.nn as nn
import mmcv
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
import numpy as np
import torch
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.backbones.swin import SwinTransformer
from mmdet3d.models.backbones.swin_bev import SwinTransformerBEVFT
from mmdet3d.models.backbones.flash_intern_image import FlashInternImage
from mmdet3d.models.alocc.heads.occ_loss_utils import CustomFocalLoss
from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import copy
import math

class VoxelevelHistoryFusion(BaseModule):
    def __init__(self,
                 vox_his_recurrence,
                 view_transformer,
                 single_bev_num_channels,
                 history_cat_conv_out_channels=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 occ_2d=False,
                 pre_gen_his_grid=True,
                 occ_size=[200,200,16],
                 with_cp=False,
                 vox_his_sup_w_his=False,
                 motion_his_fusion=False,
                 motion_his_pred_with_his=False,
                 not_use_time_emb=False,
                 max_seqlen=50,
                 motion_his_flow_bin=False,
                 motion_his_sup_w_his=False,
                 motion_his_base_lr=False,
                 motion_his_learnable_lr=False,
                 vox_pre_sup_his_length=8,
                 scene_his_mlp_mid_channel=80,
                vox_his_time_emb_long=False,
                vox_his_time_emb_fixed=False,
                vox_his_learnable_lr=False,
                vox_his_base_lr=1.,
                
                grid_config=None,
                 ):
        super().__init__()
        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                         if history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
        
        self.vox_his_recurrence=vox_his_recurrence
        self.view_transformer=view_transformer
        self.occ_2d=occ_2d

        self.pre_gen_his_grid=pre_gen_his_grid
        self.with_cp=with_cp
        if pre_gen_his_grid:
            h,w,z=occ_size
            
            if self.occ_2d:
                xs = torch.linspace(0, w - 1, w).view(1, w).expand(h, w)
                ys = torch.linspace(0, h - 1, h).view(h, 1).expand(h, w)
                self.his_grid = torch.stack(
                    (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(1, h, w, 4).view(1,h,w,1,4,1)

                # This converts BEV indices to meters
                # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
                # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
                feat2bev = torch.zeros((4,4))
                if grid_config is None:
                    
                    feat2bev[0, 0] = self.view_transformer.dx[0]
                    feat2bev[1, 1] = self.view_transformer.dx[1]
                    feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                    feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.
                else:
                    feat2bev[0, 0] = grid_config['x'][-1]
                    feat2bev[1, 1] = grid_config['y'][-1]
                    feat2bev[0, 3] = grid_config['x'][0] 
                    feat2bev[1, 3] = grid_config['y'][0]
                feat2bev[2, 2] = 1
                feat2bev[3, 3] = 1
                self.feat2bev = feat2bev.view(1,4,4)
            else:
                xs = torch.linspace(0, w - 1, w).view(1, w, 1).expand(h, w, z)
                ys = torch.linspace(0, h - 1, h).view(h, 1, 1).expand(h, w, z)
                zs = torch.linspace(0, z - 1, z).view(1, 1, z).expand(h, w, z)
                self.his_grid = torch.stack(
                    (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(1, h, w, z, 4).view(1, h, w, z, 4, 1)
                
                # This converts BEV indices to meters
                # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
                # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
                feat2bev = torch.zeros((4,4))
                if grid_config is None:
                    feat2bev[0, 0] = self.view_transformer.dx[0]
                    feat2bev[1, 1] = self.view_transformer.dx[1]
                    feat2bev[2, 2] = self.view_transformer.dx[2]
                    feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                    feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.
                    feat2bev[2, 3] = self.view_transformer.bx[2] - self.view_transformer.dx[2] / 2.
                else:
                    
                    feat2bev[0, 0] = grid_config['x'][-1]
                    feat2bev[1, 1] = grid_config['y'][-1]
                    feat2bev[2, 2] = grid_config['z'][-1]
                    feat2bev[0, 3] = grid_config['x'][0] 
                    feat2bev[1, 3] = grid_config['y'][0]
                    feat2bev[2, 3] = grid_config['z'][0] 
                # feat2bev[2, 2] = 1
                feat2bev[3, 3] = 1
                self.feat2bev = feat2bev.view(1,4,4)
        
        self.vox_his_sup_w_his=vox_his_sup_w_his
        self.motion_his_fusion=motion_his_fusion
        self.motion_his_pred_with_his=motion_his_pred_with_his
        self.not_use_time_emb=not_use_time_emb
        self.vox_his_sup_w_his=vox_his_sup_w_his
        self.max_seqlen=max_seqlen
        if motion_his_fusion:
            
            self.motion_his_pred_with_his=motion_his_pred_with_his
            if motion_his_pred_with_his:
                offset_net_input_channel=single_bev_num_channels*2
            else:
                offset_net_input_channel=single_bev_num_channels
            self.his_stream_offset_net=MotionHisoryFusion(offset_net_input_channel=offset_net_input_channel,conv=nn.Conv3d,bev_2d=occ_2d,
                                                       motion_his_flow_bin=motion_his_flow_bin,motion_his_sup_w_his=motion_his_sup_w_his,motion_his_base_lr=motion_his_base_lr,motion_his_learnable_lr=motion_his_learnable_lr,motion_dim=motion_dim,)
     
          
        
        self.max_seqlen=max_seqlen
            ## Embed each sample with its relative temporal offset with current timestep
        conv = nn.Conv2d if grid_config['z'][1]-grid_config['z'][0] ==grid_config['z'][2] else nn.Conv3d
        if vox_his_recurrence:
            self.history_cat_num=1
        
    
        self.vox_his_time_emb_long=vox_his_time_emb_long
        self.vox_his_learnable_lr=vox_his_learnable_lr
        self.vox_his_base_lr=vox_his_base_lr
        
        his_bev_channel=self.single_bev_num_channels * (self.history_cat_num + 1)
        if vox_his_learnable_lr:
            self.history_keyframe_lr_conv = nn.Sequential(
                conv(self.single_bev_num_channels * (self.history_cat_num + 1),
                        1,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.Sigmoid())
        self.vox_his_time_emb_fixed=vox_his_time_emb_fixed
        if vox_his_time_emb_fixed:
            self.time_emb_net=nn.Sequential(
                nn.Linear(self.single_bev_num_channels,self.single_bev_num_channels*2),
                nn.ReLU(inplace=True),
                nn.Linear(self.single_bev_num_channels*2,self.single_bev_num_channels))
            time_emb_dim=self.single_bev_num_channels 
        else:
            time_emb_dim=self.single_bev_num_channels + 1
        
        if vox_his_sup_w_his:
            self.history_keyframe_time_conv = nn.Sequential(
            conv(time_emb_dim,
                    self.single_bev_num_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1),
            nn.ReLU(inplace=True))
            ## Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                conv(his_bev_channel,
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.ReLU(inplace=True))
        else:
            self.history_keyframe_time_conv = nn.Sequential(
                conv(time_emb_dim,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.SyncBatchNorm(self.single_bev_num_channels),
                nn.ReLU(inplace=True))
            ## Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                conv(his_bev_channel,
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.SyncBatchNorm(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))
     
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0
                 
    @force_fp32()
    def forward_fuse_his(self, curr_bev, img_metas, bda,feat_for_pred_bias=None): # align features with 3d shift
        
        if not self.training:
            self.do_history=True
        
        if self.occ_2d:
            curr_bev = curr_bev.unsqueeze(-1)
            if feat_for_pred_bias is not None:
                feat_for_pred_bias=feat_for_pred_bias.unsqueeze(-1)
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
            if feat_for_pred_bias is not None:
                feat_for_pred_bias=feat_for_pred_bias.permute(0, 1, 4, 2, 3)
            
        
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
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1).detach()
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)
        
          
          
            if self.vox_his_sup_w_his:
                self.history_grid=[[]]*len(start_of_sequence)
                self.history_feat=[[]]*len(start_of_sequence)
                self.history_seqlen_offset=[0]*len(start_of_sequence)
                self.history_bev_his=curr_bev.repeat(1, self.history_cat_num, 1, 1, 1).detach()
            self.his_iter=torch.zeros(len(start_of_sequence)).to(curr_bev)


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
                
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1).detach()
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            
            self.his_iter[start_of_sequence]=0

        if self.vox_his_recurrence:
            
            self.history_bev=self.history_bev[:,-self.single_bev_num_channels:,...]
        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        
        for i in range(len(start_of_sequence)):
            history_bev_i_clone=None
            if self.vox_his_sup_w_his:

                if self.his_iter[i]==0:
                    self.history_grid[i]=[]
                    self.history_feat[i]=[]
                if self.his_iter[i]<=self.vox_pre_sup_his_length:
                    if len(self.history_feat[i])>0:
                        self.history_bev[i]=self.history_feat[i][0]
                
                for j in range(len(self.history_feat[i])):
                    history_feat_j=self.history_feat[i][j]
                    grid_j=self.history_grid[i][j]
              
                    
                    sampled_history_bev,_,tmp_bev,mc,normalize_factor=self.warp_his_(history_feat_j,self.history_bev[i:i+1],forward_augs[i:i+1],curr_to_prev_ego_rt[i:i+1],voxel_feat,start_of_sequence,feat_for_pred_bias,grid=grid_j)
                    
                    if self.vox_his_time_emb_long:
                        history_sweep_time=torch.Tensor([[j,j-1]]).to(curr_bev).clamp(0)
                    else:
                        history_sweep_time=torch.Tensor([[0.,1]]).to(curr_bev)
                    feats_to_return,_=self.his_net_func(history_feat_j, sampled_history_bev,voxel_feat,1, mc, z, h, w,c_,torch.Tensor([[0.,1]]).to(curr_bev))
                    
                    if self.occ_2d:
                        feats_to_return=feats_to_return.unsqueeze(2)
                    
                    history_bev_=self.history_bev.clone()
                    history_bev_[i:i+1]=feats_to_return
                    self.history_bev=history_bev_
                    
                    if j==0 and self.his_iter[i]>=self.vox_pre_sup_his_length:
                        history_bev_i_clone=self.history_bev[i].detach().clone()
    
                
                
        sampled_history_bev,grid,tmp_bev,mc,normalize_factor,rt_flow=self.warp_his_(curr_bev,self.history_bev,forward_augs,curr_to_prev_ego_rt,voxel_feat,start_of_sequence,feat_for_pred_bias)
        
        for i in range(len(start_of_sequence)):
            if self.vox_his_sup_w_his:
                self.history_grid[i].append(grid[i:i+1])
                self.history_feat[i].append(curr_bev[i:i+1].detach())
                if len(self.history_grid)>self.vox_pre_sup_his_length:
                    self.history_grid=self.history_grid[1:]
                    self.history_feat=self.history_feat[1:]
        
        

        ## Update history
        # Add in current frame to features & timestep
        results={}

        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)
        
        
        if self.vox_his_recurrence:
            self.history_sweep_time=self.history_sweep_time[:,:2]
            
        if self.vox_his_time_emb_long:
            history_sweep_time=torch.stack((self.his_iter,self.his_iter-1),dim=0).permute(1,0).clamp(0)
        else:
            history_sweep_time=self.history_sweep_time
        feats_to_return,feats_cat=self.his_net_func(curr_bev, sampled_history_bev,voxel_feat,n, mc, z, h, w,c_,history_sweep_time,additional_his_bev=feats_to_return)
          
        # Time conv
        if self.occ_2d:
            feats_to_return=feats_to_return.unsqueeze(2)
        if self.vox_his_recurrence:
            self.history_bev = feats_to_return.detach().clone()
        else:
            self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach()#.clone()
       
        if history_bev_i_clone is not None:
            self.history_bev[i]=history_bev_i_clone
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if self.vox_his_recurrence:
            if start_of_sequence.sum()>0:
                tem=feats_to_return.clone()
                tem[start_of_sequence] = curr_bev[start_of_sequence].clone()+feats_to_return[start_of_sequence]*0.
                feats_to_return=tem
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        
        if self.occ_2d:
            feats_to_return=feats_to_return.squeeze(-1)
        results.update({'fused_bev_feat':feats_to_return.clone()})
        self.his_iter+=1
        
        return results
    def warp_his_(self,curr_bev,tmp_bev,forward_augs,curr_to_prev_ego_rt,voxel_feat,start_of_sequence,feat_for_pred_bias,normalize_factor=None,rt_flow=None,grid=None):
        n, c_, z, h, w=curr_bev.shape
        if not self.occ_2d:
            if grid is None:
                if self.pre_gen_his_grid:
                    grid=self.his_grid.to(curr_bev.device).repeat(n,1,1,1,1,1,)
                    feat2bev=self.feat2bev.to(curr_bev.device)
                else:
                    # Generate grid
                    xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
                    ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
                    zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
                    grid = torch.stack(
                        (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
                    # This converts BEV indices to meters
                    # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
                    # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
                    feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
                    # feat2bev[0, 0] = self.view_transformer.dx[0]
                    # feat2bev[1, 1] = self.view_transformer.dx[1]
                    # feat2bev[2, 2] = self.view_transformer.dx[2]
                    # feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                    # feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.
                    # feat2bev[2, 3] = self.view_transformer.bx[2] - self.view_transformer.dx[2] / 2.

                    feat2bev[0, 0] = self.grid_config['x'][2]
                    feat2bev[1, 1] = self.grid_config['y'][2]
                    feat2bev[2, 2] = self.grid_config['z'][2]
                    feat2bev[0, 3] = self.grid_config['x'][0]
                    feat2bev[1, 3] = self.grid_config['y'][0]
                    feat2bev[2, 3] = self.grid_config['z'][0]
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
            
            
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            if self.motion_his_fusion:
                rt_flow2=rt_flow
                # self.history_sample_bias=torch.tensor([[0,0],[0,4],[0,-4],[4,0],[-4,0]])
                # conv = nn.Conv2d if self.view_transformer.nx[-1] == 1 else nn.Conv3d
                if feat_for_pred_bias is None:
                    feat_for_pred_bias=curr_bev
                if self.motion_his_pred_with_his:
                    sampled_history_bev_wo_offset = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)
                    feat_for_pred_bias=torch.cat((sampled_history_bev_wo_offset,curr_bev),dim=1)
     
                sampled_history_bev,grid=self.his_stream_offset_net(feat_for_pred_bias,tmp_bev,grid,normalize_factor,rt_flow2,start_of_sequence=start_of_sequence)
            
            else:
                sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)
        
        else:
            if grid is None:
                if self.pre_gen_his_grid:
                    grid=self.his_grid.to(curr_bev.device).repeat(n,1,1,1,1,1,)
                    feat2bev=self.feat2bev.to(curr_bev.device)
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
                    # feat2bev[0, 0] = self.view_transformer.dx[0]
                    # feat2bev[1, 1] = self.view_transformer.dx[1]
                    # feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                    # feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.

                    feat2bev[0, 0] = self.grid_config['x'][2]
                    feat2bev[1, 1] = self.grid_config['y'][2]
                    feat2bev[0, 3] = self.grid_config['x'][0]
                    feat2bev[1, 3] = self.grid_config['y'][0]
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
            
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            if self.motion_his_fusion:
                # self.history_sample_bias=torch.tensor([[0,0],[0,4],[0,-4],[4,0],[-4,0]])
                # conv = nn.Conv2d if self.view_transformer.nx[-1] == 1 else nn.Conv3d
                rt_flow2=rt_flow
                if feat_for_pred_bias is None:
                    feat_for_pred_bias=curr_bev
                if self.motion_his_pred_with_his:
                    sampled_history_bev_wo_offset = F.grid_sample(tmp_bev[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)
                    sampled_history_bev_wo_offset=sampled_history_bev_wo_offset.unsqueeze(2)
                    feat_for_pred_bias=torch.cat((sampled_history_bev_wo_offset,curr_bev),dim=1)
     
                sampled_history_bev,grid=self.his_stream_offset_net(feat_for_pred_bias,tmp_bev,grid,normalize_factor,rt_flow2,start_of_sequence=start_of_sequence)
                
            else:
                sampled_history_bev = F.grid_sample(tmp_bev[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)
         
        return sampled_history_bev,grid,tmp_bev,mc,normalize_factor,rt_flow
    def his_net_func(self,curr_bev, sampled_history_bev,voxel_feat,n, mc, z, h, w,c_,history_sweep_time,additional_his_bev=None):
        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 
        
        if self.vox_his_recurrence:
            feats_to_return = feats_cat.reshape(feats_cat.shape[0], 1+ 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        else:
            feats_to_return = feats_cat.reshape(
                    feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if not self.not_use_time_emb:
           
            if voxel_feat:
               
                if self.vox_his_time_emb_fixed:
                    time_emb=positional_encoding_continual_1d(self.single_bev_num_channels,history_sweep_time*self.history_cam_sweep_freq)
                    _,n_his,_=time_emb.shape
                    time_emb=time_emb.reshape(n*n_his,-1)
                    time_emb=self.time_emb_net(time_emb)
                    time_emb=time_emb.reshape(n,n_his,-1)
                    feats_to_return=feats_to_return+time_emb[:,:,:,None,None, None].repeat(1, 1, 1, *feats_to_return.shape[3:])
                else:
                    feats_to_return = torch.cat(
                    [feats_to_return, history_sweep_time[:, :, None, None, None, None].repeat(1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
                    ], dim=2) # B x (1 + T) x 81 x Z x H x W
            else:
                feats_to_return = torch.cat(
                [feats_to_return, history_sweep_time[:, :, None, None, None].repeat(
                    1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
                ], dim=2) # B x (1 + T) x 81 x H x W

        if self.occ_2d:
            feats_to_return=feats_to_return.squeeze(3)

        if self.with_cp:
            feats_to_return = checkpoint(self.history_keyframe_time_conv, feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:])
        else:
            feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        if self.vox_his_learnable_lr:
            lr=self.history_keyframe_lr_conv(feats_to_return.reshape(feats_to_return.shape[0], -1, *feats_to_return.shape[3:]))*self.vox_his_base_lr
            feats_to_return1,feats_to_return2=feats_to_return[:,0:1,...],feats_to_return[:,1:2,...]
            weight1,weight2=self.history_keyframe_cat_conv[0].weight[:,:feats_to_return1.shape[2]],self.history_keyframe_cat_conv[0].weight[:,feats_to_return1.shape[2]:feats_to_return1.shape[2]*2]
            
            if self.occ_2d:
                feats_to_return1_weighted=torch.einsum('ijhw,bcihw->bcjhw',weight1,feats_to_return1).squeeze(1)
                feats_to_return2_weighted=torch.einsum('ijhw,bcihw->bcjhw',weight2,feats_to_return2).squeeze(1)
                bias=self.history_keyframe_cat_conv[0].bias[:,None,None]
            else:
                feats_to_return1_weighted=torch.einsum('ijzhw,bcizhw->bcjzhw',weight1,feats_to_return1).squeeze(1)
                feats_to_return2_weighted=torch.einsum('ijzhw,bcizhw->bcjzhw',weight2,feats_to_return2).squeeze(1)
                bias=self.history_keyframe_cat_conv[0].bias[:,None,None,None]
            feats_to_return=feats_to_return2.squeeze(1)-feats_to_return2_weighted*lr+feats_to_return1_weighted*lr+bias
            
            feats_to_return=self.history_keyframe_cat_conv[1](feats_to_return)
            feats_to_return=self.history_keyframe_cat_conv[2](feats_to_return)
        else:      
            if self.with_cp:
                
                feats_to_return =checkpoint(self.history_keyframe_cat_conv,feats_to_return.reshape(feats_to_return.shape[0], -1, *feats_to_return.shape[3:]))
            else:
                feats_to_return = self.history_keyframe_cat_conv(
                    feats_to_return.reshape(feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        return feats_to_return,feats_cat
    
    

class SceneLevelHistoryFusion(BaseModule):

    def __init__(self,
                 scene_his_learned_lr,
                 scene_his_base_lr,
                 single_bev_num_channels,
                 scene_his_post_project=False,
                 scene_his_pre_detach=False,
                 scene_his_warm_up_start=0.1,
                 scene_his_warm_up_iter=4,
                 squential_length=20,
                 scene_his_warm_up=False,
                 scene_his_sup_w_his=False,
                 scene_his_with_virtual_adverse=False,
                 scene_his_with_virtual_adverse_weight=0.1,
                 scene_his_decay=False,
                 scene_his_decay_rate=0.1,
                 scene_his_learnable_decay=False,
                 scene_his_mlp=False,
                 scene_his_mlp_mid_channel=80,
                 ):
        super().__init__()
    
   
        if not scene_his_mlp:
            scene_his_mlp_mid_channel=single_bev_num_channels

        self.scene_his_q=nn.Linear(single_bev_num_channels,single_bev_num_channels,bias=False)
        self.scene_his_k=nn.Linear(single_bev_num_channels,single_bev_num_channels,bias=False)
        self.scene_his_v=nn.Linear(single_bev_num_channels,single_bev_num_channels,bias=False)
    
            
        self.scene_his_learnable_weight=nn.Parameter(torch.randn(1,scene_his_mlp_mid_channel,single_bev_num_channels)*0.02)
        if scene_his_learned_lr:
            self.scene_his_learned_lr=nn.Parameter(torch.randn(1,1,single_bev_num_channels)*0.02)
            self.scene_his_learned_lr_bias=nn.Parameter(torch.zeros(1))
        
        
            
        self.scene_his_w=None
        self.scene_his_w_grad=None
        self.scene_his_base_lr=scene_his_base_lr
        self.scene_his_learned_lr=scene_his_learned_lr
        
        self.scene_his_learnable_bias=nn.Parameter(torch.zeros(1,scene_his_mlp_mid_channel,1))
        
        
    
        ln_weight_data = nn.LayerNorm(single_bev_num_channels).weight.data
        self.scene_his_norm_weight = nn.Parameter(ln_weight_data.unsqueeze(0).unsqueeze(-1))
        ln_bias_data = nn.LayerNorm(single_bev_num_channels).bias.data
        self.scene_his_norm_bias = nn.Parameter(ln_bias_data.unsqueeze(0).unsqueeze(-1))
    
        
        if scene_his_post_project:
            self.scene_his_out_project=nn.Linear(single_bev_num_channels,single_bev_num_channels,bias=False)
            self.scene_his_post_norm = nn.LayerNorm(single_bev_num_channels, eps=1e-6)
    
        if scene_his_mlp:
            self.scene_his_learnable_weight2=nn.Parameter(torch.randn(1,single_bev_num_channels,scene_his_mlp_mid_channel)*0.02)
            self.scene_his_learnable_bias2=nn.Parameter(torch.zeros(1,single_bev_num_channels,1))
    
    
        self.scene_his_post_project=scene_his_post_project
        self.scene_his_pre_detach=scene_his_pre_detach

        self.scene_his_warm_up=scene_his_warm_up
        if scene_his_warm_up:
            self.scene_his_warm_up_start=scene_his_warm_up_start
            self.scene_his_warm_up_iter=scene_his_warm_up_iter
            self.squential_length=squential_length

        self.scene_his_sup_w_his=scene_his_sup_w_his
        if scene_his_sup_w_his:
            self.scene_his_feat=None
        
        self.scene_his_with_virtual_adverse=scene_his_with_virtual_adverse
        self.scene_his_with_virtual_adverse_weight=scene_his_with_virtual_adverse_weight
        self.scene_his_decay=scene_his_decay
        self.scene_his_decay_rate=scene_his_decay_rate
        self.scene_his_learnable_decay=scene_his_learnable_decay
        if scene_his_learnable_decay:
            self.scene_his_learnable_decay_net=nn.Sequential(
                nn.Linear(single_bev_num_channels,single_bev_num_channels,bias=False),
                nn.ReLU(),
                nn.Linear(single_bev_num_channels,1,bias=False),
                nn.Sigmoid(),
            )
  
        self.scene_his_mlp=scene_his_mlp
       
    # https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py  
    def ln_fwd(self,x, gamma, beta,dim=None, eps=1e-6):
        "Batch forward for LayerNorm."

        # Mean and variance computation
        mu = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)

        # Normalization
        std = torch.sqrt(var + eps)
        x_hat = (x - mu) / std

        # Scale and shift
        y = gamma * x_hat + beta

        return y
    def ln_fused_l2_bwd(self,x, l2_target, gamma, beta, dim=None,eps=1e-6):
        "Batch backward for LayerNorm fused with L2 loss."
        D = x.shape[dim]

        # Mean and variance computation
        mu = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)

        # Normalization
        std = torch.sqrt(var + eps)
        x_hat = (x - mu) / std

        # Scale and shift
        y = gamma * x_hat + beta
        grad_output = y - l2_target
        grad_x_hat = grad_output * gamma
        z = (
            (1.0 / D)
            * (
                D * grad_x_hat
                - grad_x_hat.sum(dim=dim, keepdim=True)
                - x_hat * (grad_x_hat * x_hat).sum(dim=dim, keepdim=True)
            )
            / std
        )

        return z,grad_output,x_hat
    # Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
    def gelu_bwd(self,x):
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        return ff
    def sync_param(self,param):
        if not param.is_contiguous():
            param = param.contiguous()

        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)

        param.data /= dist.get_world_size()
  
    def forward(self,x,start_of_sequence):
        ##
        #x #b,d,k
        ##
     
        
        b,_,_=x.shape
        if self.scene_his_w_grad is None:
            self.scene_his_w_grad=torch.zeros_like(self.scene_his_learnable_weight.repeat(b,1,1),device=x.device)
            self.scene_his_b_grad=torch.zeros_like(self.scene_his_learnable_bias.repeat(b,1,1),device=x.device)
            
            if self.scene_his_mlp:
                self.scene_his_w_grad2=torch.zeros_like(self.scene_his_learnable_weight2.repeat(b,1,1),device=x.device)
                self.scene_his_b_grad2=torch.zeros_like(self.scene_his_learnable_bias2.repeat(b,1,1),device=x.device)
                
       
            self.scene_his_gamma_grad=torch.zeros_like(self.scene_his_norm_weight.repeat(b,1,1),device=x.device)
            self.scene_his_beta_grad=torch.zeros_like(self.scene_his_norm_bias.repeat(b,1,1),device=x.device)
            
            self.scene_his_iter=torch.zeros(b,1,device=x.device)
            
            if self.scene_his_sup_w_his:
                self.scene_his_feat=[[]]*b
                
            

        if sum(start_of_sequence)>0:
            self.scene_his_w_grad[start_of_sequence]=0
            self.scene_his_b_grad[start_of_sequence]=0
            self.scene_his_iter[start_of_sequence]=0
            
            if self.scene_his_mlp:
                self.scene_his_w_grad2[start_of_sequence]=0
                self.scene_his_b_grad2[start_of_sequence]=0
            
      
            self.scene_his_gamma_grad[start_of_sequence]=0
            self.scene_his_beta_grad[start_of_sequence]=0
            
            if self.scene_his_sup_w_his:
                for i in range(len(start_of_sequence)):
                    if start_of_sequence[i]:
                        self.scene_his_feat[i]=[]
            
              

        q=self.scene_his_q.weight.unsqueeze(0)
        k=self.scene_his_k.weight.unsqueeze(0)
        v=self.scene_his_v.weight.unsqueeze(0)

        scene_his_w=self.scene_his_learnable_weight-self.scene_his_w_grad
        scene_his_b=self.scene_his_learnable_bias-self.scene_his_b_grad
        if self.scene_his_mlp:
            ttt_w2=self.scene_his_learnable_weight2-self.scene_his_w_grad2
            ttt_b2=self.scene_his_learnable_bias2-self.scene_his_b_grad2
            
  
        scene_his_gamma=self.scene_his_norm_weight-self.scene_his_gamma_grad
        scene_his_beta=self.scene_his_norm_bias-self.scene_his_beta_grad
 
        if self.scene_his_pre_detach and self.training:
            scene_his_w=scene_his_w.detach().clone()
            scene_his_b=scene_his_b.detach().clone()
            
            if self.scene_his_mlp:
                ttt_w2=ttt_w2.detach().clone()
                ttt_b2=ttt_b2.detach().clone()
                

            scene_his_gamma=scene_his_gamma.detach().clone()
            scene_his_beta=scene_his_beta.detach().clone()

        
        if self.scene_his_sup_w_his:
            for i in range(b):
                
                for j in range(int(self.scene_his_iter[i])):
                    
                    
                    if self.scene_his_decay:
                        if self.scene_his_learnable_decay:
                            
                            scene_his_decay_rate=self.scene_his_learnable_decay_net(self.scene_his_feat[i][j].mean(-1)).unsqueeze(-1)
                            
                        else:
                            scene_his_decay_rate=self.scene_his_decay_rate
                        if self.scene_his_mlp:
                            results_ij=self.mlp_forward(self.scene_his_feat[i][j],k,v,
                                                 self.scene_his_learnable_weight-self.scene_his_w_grad[i:i+1]*scene_his_decay_rate,
                                                 self.scene_his_learnable_bias-self.scene_his_b_grad[i:i+1]*scene_his_decay_rate,
                                                 self.scene_his_learnable_weight2-self.scene_his_w_grad2[i:i+1]*scene_his_decay_rate,
                                                 self.scene_his_learnable_bias2-self.scene_his_b_grad2[i:i+1]*scene_his_decay_rate)
                            w_grad_ij,b_grad_ij,w_grad2_ij,b_grad2_ij,lr=results_ij['w_grad'],results_ij['b_grad'],results_ij['w_grad2'],results_ij['b_grad2'],results_ij['lr']
                   
                        else:
                            results_ij=\
                                self.linear_forward(self.scene_his_feat[i][j],k,v,
                                                 self.scene_his_learnable_weight-self.scene_his_w_grad[i:i+1]*scene_his_decay_rate,
                                                 self.scene_his_learnable_bias-self.scene_his_b_grad[i:i+1]*scene_his_decay_rate)
                            w_grad_ij,b_grad_ij,lr=results_ij['w_grad'],results_ij['b_grad'],results_ij['lr']

                        scene_his_w_grad_=self.scene_his_w_grad.clone()
                        scene_his_b_grad_=self.scene_his_b_grad.clone()
                        scene_his_w_grad_[i:i+1]=self.scene_his_w_grad[i:i+1]*scene_his_decay_rate+w_grad_ij*lr*(1-scene_his_decay_rate)
                        scene_his_b_grad_[i:i+1]=self.scene_his_b_grad[i:i+1]*scene_his_decay_rate+b_grad_ij*lr*(1-scene_his_decay_rate)

                        self.scene_his_w_grad=scene_his_w_grad_
                        self.scene_his_b_grad=scene_his_b_grad_
                        
                        if self.scene_his_mlp:
                            scene_his_w_grad2_=self.scene_his_w_grad2.clone()
                            scene_his_b_grad2_=self.scene_his_b_grad2.clone()
                            scene_his_w_grad2_[i:i+1]=self.scene_his_w_grad2[i:i+1]*scene_his_decay_rate+w_grad2_ij*lr*(1-scene_his_decay_rate)
                            scene_his_b_grad2_[i:i+1]=self.scene_his_b_grad2[i:i+1]*scene_his_decay_rate+b_grad2_ij*lr*(1-scene_his_decay_rate)

                            self.scene_his_w_grad2=scene_his_w_grad2_
                            self.scene_his_b_grad2=scene_his_b_grad2_
                    
                        scene_his_gamma_grad_=self.scene_his_gamma_grad.clone()
                        scene_his_beta_grad_=self.scene_his_beta_grad.clone()
                        scene_his_gamma_grad_[i:i+1]=self.scene_his_gamma_grad[i:i+1]*scene_his_decay_rate+results_ij['gamma_grad']*lr*(1-scene_his_decay_rate)
                        scene_his_beta_grad_[i:i+1]=self.scene_his_beta_grad[i:i+1]*scene_his_decay_rate+results_ij['beta_grad']*lr*(1-scene_his_decay_rate)
                        self.scene_his_gamma_grad=scene_his_gamma_grad_
                        self.scene_his_beta_grad=scene_his_beta_grad_
                    else:   
                        if self.scene_his_mlp:
                            results_ij=self.mlp_forward(self.scene_his_feat[i][j],k,v,
                                                 self.scene_his_learnable_weight-self.scene_his_w_grad[i:i+1],
                                                 self.scene_his_learnable_bias-self.scene_his_b_grad[i:i+1],
                                                 self.scene_his_learnable_weight2-self.scene_his_w_grad2[i:i+1],
                                                 self.scene_his_learnable_bias2-self.scene_his_b_grad2[i:i+1])
                            w_grad_ij,b_grad_ij,w_grad2_ij,b_grad2_ij,lr=results_ij['w_grad'],results_ij['b_grad'],results_ij['w_grad2'],results_ij['b_grad2'],results_ij['lr']
    
                        else:
                            results_ij=\
                                self.linear_forward(self.scene_his_feat[i][j],k,v,
                                                 self.scene_his_learnable_weight-self.scene_his_w_grad[i:i+1],
                                                 self.scene_his_learnable_bias-self.scene_his_b_grad[i:i+1])
                            w_grad_ij,b_grad_ij,lr=results_ij['w_grad'],results_ij['b_grad'],results_ij['lr']
                        self.scene_his_w_grad[i:i+1]=self.scene_his_w_grad[i:i+1]+w_grad_ij*lr
                        self.scene_his_b_grad[i:i+1]=self.scene_his_b_grad[i:i+1]+b_grad_ij*lr
                        
                        if self.scene_his_mlp:
                            self.scene_his_w_grad2[i:i+1]=self.scene_his_w_grad2[i:i+1]+w_grad2_ij*lr
                            self.scene_his_b_grad2[i:i+1]=self.scene_his_b_grad2[i:i+1]+b_grad2_ij*lr
                  
                        self.scene_his_gamma_grad[i:i+1]=self.scene_his_gamma_grad[i:i+1]+results_ij['gamma_grad']*lr
                        self.scene_his_beta_grad[i:i+1]=self.scene_his_beta_grad[i:i+1]+results_ij['beta_grad']*lr
   
                self.scene_his_feat[i].append(x[i:i+1].detach())
            scene_his_w=self.scene_his_learnable_weight-self.scene_his_w_grad
            scene_his_b=self.scene_his_learnable_bias-self.scene_his_b_grad
            if self.scene_his_mlp:
                ttt_w2=self.scene_his_learnable_weight2-self.scene_his_w_grad2
                ttt_b2=self.scene_his_learnable_bias2-self.scene_his_b_grad2

            scene_his_gamma=self.scene_his_norm_weight-self.scene_his_gamma_grad
            scene_his_beta=self.scene_his_norm_bias-self.scene_his_beta_grad
                
        if not self.scene_his_mlp:   
            
            results=self.linear_forward(x,k,v,scene_his_w,scene_his_b,scene_his_gamma,scene_his_beta)
            w_grad,b_grad,lr=results['w_grad'],results['b_grad'],results['lr']
        else:
            results=self.mlp_forward(x,k,v,scene_his_w,scene_his_b,ttt_w2,ttt_b2,scene_his_gamma,scene_his_beta)
            w_grad,b_grad,w_grad2,b_grad2,lr=results['w_grad'],results['b_grad'],results['w_grad2'],results['b_grad2'],results['lr']
        

        gamma_grad,beta_grad=results['gamma_grad'],results['beta_grad']
        
        w_grad=w_grad*lr
        b_grad=b_grad*lr
        
        if self.scene_his_mlp:     
            w_grad2=w_grad2*lr
            b_grad2=b_grad2*lr

        gamma_grad=gamma_grad*lr
        beta_grad=beta_grad*lr

        
        if self.scene_his_decay:
            if self.scene_his_learnable_decay:
                
                scene_his_decay_rate=self.scene_his_learnable_decay_net(x.mean(-1)).unsqueeze(-1)
                
            else:
                scene_his_decay_rate=self.scene_his_decay_rate
            
            scene_his_w_grad=self.scene_his_w_grad*scene_his_decay_rate+w_grad*(1-scene_his_decay_rate)
            scene_his_b_grad=self.scene_his_b_grad*scene_his_decay_rate+b_grad*(1-scene_his_decay_rate)
            if self.scene_his_mlp:
                scene_his_w_grad2=self.scene_his_w_grad2*scene_his_decay_rate+w_grad2*(1-scene_his_decay_rate)
                scene_his_b_grad2=self.scene_his_b_grad2*scene_his_decay_rate+b_grad2*(1-scene_his_decay_rate)
  
            scene_his_gamma_grad=self.scene_his_gamma_grad*scene_his_decay_rate+gamma_grad*(1-scene_his_decay_rate)
            scene_his_beta_grad=self.scene_his_beta_grad*scene_his_decay_rate+beta_grad*(1-scene_his_decay_rate)
        else:
            scene_his_w_grad=self.scene_his_w_grad+w_grad
            scene_his_b_grad=self.scene_his_b_grad+b_grad
            if self.scene_his_mlp:
                scene_his_w_grad2=self.scene_his_w_grad2+w_grad2
                scene_his_b_grad2=self.scene_his_b_grad2+b_grad2

            scene_his_gamma_grad=self.scene_his_gamma_grad+gamma_grad
            scene_his_beta_grad=self.scene_his_beta_grad+beta_grad
        scene_his_w=self.scene_his_learnable_weight-scene_his_w_grad
        scene_his_b=self.scene_his_learnable_bias-scene_his_b_grad
        
        if self.scene_his_mlp:
            ttt_w2=self.scene_his_learnable_weight2-scene_his_w_grad2
            ttt_b2=self.scene_his_learnable_bias2-scene_his_b_grad2

        scene_his_gamma=self.scene_his_norm_weight-scene_his_gamma_grad
        scene_his_beta=self.scene_his_norm_bias-scene_his_beta_grad

        # out=self.scene_his_w.matmul(q)

        qx=q.matmul(x)
        
        Z1=scene_his_w.matmul(qx)+scene_his_b
        if not self.scene_his_mlp:
            out=Z1
        else:
            X2 = F.gelu(Z1, approximate="tanh")
            out=ttt_w2.matmul(X2)+ttt_b2
        
        self.scene_his_w_grad=scene_his_w_grad.detach().clone()

        self.scene_his_b_grad=scene_his_b_grad.detach().clone()
    
        if  self.scene_his_mlp:
            self.scene_his_w_grad2=scene_his_w_grad2.detach().clone()

            self.scene_his_b_grad2=scene_his_b_grad2.detach().clone()
            
        out=self.ln_fwd(out, scene_his_gamma,scene_his_beta,dim=1, eps=1e-6)+qx
        if self.scene_his_post_project:
            out=out.permute(0,2,1)
            out=self.scene_his_post_norm(out)
            
            out=self.scene_his_out_project(out)
            out=out.permute(0,2,1)
   
        self.scene_his_iter+=1
        return out    
    def linear_forward(self,x,k,v,scene_his_w,scene_his_b,scene_his_gamma,scene_his_beta):
        if self.scene_his_with_virtual_adverse:
            x_grad=[]
            for i in range(x.shape[0]):
                with torch.enable_grad():

                    x_i=x[i].detach().clone().requires_grad_(True)
                    ttt_w_i=scene_his_w[i].detach()
                    ttt_b_i=scene_his_b[i].detach()

                    kxi=k.detach().matmul(x_i)
                    Z1=ttt_w_i.matmul(kxi)+ttt_b_i
                    Z1_norm=self.ln_fwd(Z1, scene_his_gamma.squeeze(0), scene_his_beta.squeeze(0),dim=0, eps=1e-6)
                    
                    inner_loss=(kxi+Z1_norm-v.detach().matmul(x_i)).pow(2).mean()
                    x_grad_i=torch.autograd.grad(inner_loss,[x_i],create_graph=False)[0]
                    x_grad_i=F.normalize(x_grad_i,dim=0)
        
                    x_grad.append(x_grad_i.detach())
            
            x_grad=torch.stack(x_grad)
            x_aug=x+x_grad*self.scene_his_with_virtual_adverse_weight
        else:
            x_aug=x
    
        kx=k.matmul(x_aug)
        Z1=scene_his_w.matmul(kx)+scene_his_b
        reconstruction_target=v.matmul(x)-kx
                
        grad_l_wrt_Z1,grad_l_wrt_beta,Z1_norm = self.ln_fused_l2_bwd(Z1, reconstruction_target,scene_his_gamma, scene_his_beta,dim=1)
        grad_l_wrt_Z1=grad_l_wrt_Z1*2
        grad_l_wrt_beta=grad_l_wrt_beta*2

        w_grad=grad_l_wrt_Z1.matmul(kx.permute(0,2,1))/grad_l_wrt_Z1.shape[-1]/grad_l_wrt_Z1.shape[-2]
        
        b_grad=grad_l_wrt_Z1.sum(-1,keepdim=True)/grad_l_wrt_Z1.shape[-1]/grad_l_wrt_Z1.shape[-2]
        
        
        gamma_grad=(Z1_norm*grad_l_wrt_beta).mean(-1,keepdim=True)/grad_l_wrt_beta.shape[-2]
        beta_grad=grad_l_wrt_beta.mean(-1,keepdim=True)/grad_l_wrt_beta.shape[-2]
            
        if self.scene_his_warm_up:
            
            lr=self.scene_his_iter.clamp(max=self.scene_his_warm_up_iter)/self.scene_his_warm_up_iter*(self.scene_his_base_lr-self.scene_his_warm_up_start)+self.scene_his_warm_up_start
            lr=lr.unsqueeze(-1)
        else:
            lr=self.scene_his_base_lr
        
            
        if not self.scene_his_learned_lr:
            lr=lr
        else:
            
            lr=lr*(self.scene_his_learned_lr.matmul(x.mean(-1,keepdim=True))+self.scene_his_learned_lr_bias).sigmoid()
        
        results={}
        results['w_grad'],results['b_grad'],results['lr']=w_grad,b_grad,lr
   
        results['gamma_grad']=gamma_grad
        results['beta_grad']=beta_grad
        return results
    def mlp_forward(self,x,k,v,scene_his_w,scene_his_b,ttt_w2,ttt_b2,scene_his_gamma,scene_his_beta):
      
        if self.scene_his_with_virtual_adverse:
            ##
            #not implemented for mlp
            ##
            x_grad=[]
            for i in range(x.shape[0]):
                
                with torch.enable_grad():
                    # ttt_w_i=scene_his_w[i]
                    x_i=x[i].detach().clone().requires_grad_(True)
                    ttt_w_i=scene_his_w[i].detach()
                    ttt_b_i=scene_his_b[i].detach()
                    
                    kxi=k.detach().matmul(x_i)
                    Z1=ttt_w_i.matmul(kxi)+ttt_b_i
                    
                    Z1_norm=self.ln_fwd(Z1, scene_his_gamma.squeeze(0), scene_his_beta.squeeze(0),dim=0, eps=1e-6)
            
                    inner_loss=(kxi+Z1_norm-v.detach().matmul(x_i)).pow(2).mean()
                    
                
                    x_grad_i=torch.autograd.grad(inner_loss,[x_i],create_graph=False)[0]
                    
                    x_grad_i=F.normalize(x_grad_i,dim=0)
                    
                    x_grad.append(x_grad_i.detach())
            
            x_grad=torch.stack(x_grad)
            x_aug=x+x_grad*self.scene_his_with_virtual_adverse_weight
        else:
            x_aug=x

        kx=k.matmul(x_aug)
        Z1=scene_his_w.matmul(kx)+scene_his_b
        # X2=Z1
        X2=F.gelu(Z1, approximate="tanh")
        
        Z2 = ttt_w2.matmul(X2)+ttt_b2
        reconstruction_target=v.matmul(x)-kx
                
        grad_l_wrt_Z2,grad_l_wrt_beta,Z2_norm = self.ln_fused_l2_bwd(Z2, reconstruction_target,scene_his_gamma,scene_his_beta,dim=1)
        grad_l_wrt_Z2=grad_l_wrt_Z2*2
        grad_l_wrt_beta=grad_l_wrt_beta*2
        
        grad_l_wrt_Z1 = ttt_w2.permute(0,2,1)@grad_l_wrt_Z2  *self.gelu_bwd( Z1)*2
        
        w_grad=grad_l_wrt_Z1.matmul(kx.permute(0,2,1))/grad_l_wrt_Z1.shape[-1]/grad_l_wrt_Z1.shape[-2]
        
        b_grad=grad_l_wrt_Z1.sum(-1,keepdim=True)/grad_l_wrt_Z1.shape[-1]/grad_l_wrt_Z1.shape[-2]
        
        
        w_grad2=grad_l_wrt_Z2.matmul(X2.permute(0,2,1))/grad_l_wrt_Z2.shape[-1]/grad_l_wrt_Z2.shape[-2]
        b_grad2=grad_l_wrt_Z2.sum(-1,keepdim=True)/grad_l_wrt_Z2.shape[-1]/grad_l_wrt_Z2.shape[-2]
        

        gamma_grad=(Z2_norm*grad_l_wrt_beta).mean(-1,keepdim=True)/grad_l_wrt_beta.shape[-2]
        beta_grad=grad_l_wrt_beta.mean(-1,keepdim=True)/grad_l_wrt_beta.shape[-2]
            
        
        if self.scene_his_warm_up:
            ###
            #not implemented for mlp
            ##
            lr=self.scene_his_iter.clamp(max=self.scene_his_warm_up_iter)/self.scene_his_warm_up_iter*(self.scene_his_base_lr-self.scene_his_warm_up_start)+self.scene_his_warm_up_start
            lr=lr.unsqueeze(-1)
        else:
            lr=self.scene_his_base_lr
        if not self.scene_his_learned_lr:
            lr=lr
        else:
            lr=lr*(self.scene_his_learned_lr.matmul(x.mean(-1,keepdim=True))+self.scene_his_learned_lr_bias).sigmoid()
        
        results={}
        results['w_grad'],results['b_grad'],results['w_grad2'],results['b_grad2'],results['lr']=w_grad,b_grad,w_grad2,b_grad2,lr

        results['gamma_grad']=gamma_grad
        results['beta_grad']=beta_grad
        return results
    def save_graph(self,grad):
        global saved_graph
        saved_graph=grad.grad_fn
        return grad

class MotionHisoryFusion(BaseModule):

    def __init__(self,
                 offset_net_input_channel,
                 conv,
                 bev_2d,
                 motion_his_sup_w_his=False,
                 motion_his_flow_bin=False,
                 bin_invertal=[-22,22,1],
                 motion_his_base_lr=1.,
                 motion_his_learnable_lr=False,
                 motion_dim=2,
                
                 ):
        super().__init__()
        self.motion_his_flow_bin=motion_his_flow_bin
        
        if motion_his_flow_bin:
            self.bin_invertal=bin_invertal
            self.offset_bin=torch.from_numpy(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))+0.5
            self.history_stream_offset_net = conv(offset_net_input_channel ,
                            len(self.offset_bin)*2,
                            kernel_size=1,
                            padding=0,
                            stride=1,)
        else:
            self.history_stream_offset_net = conv(offset_net_input_channel ,
                        motion_dim,
                        kernel_size=1,
                        padding=0,
                        stride=1)
        
        self.history_sample_bias=torch.zeros(motion_dim)  
        self.bev_2d=bev_2d    
        self.offset_his= None
        self.motion_his_sup_w_his=motion_his_sup_w_his
        self.motion_his_base_lr=motion_his_base_lr
        if motion_his_learnable_lr:
            self.offset_lr_net = conv(offset_net_input_channel ,
                            1,
                            kernel_size=1,
                            padding=0,
                            stride=1)
        self.motion_his_learnable_lr=motion_his_learnable_lr
        self.motion_dim=motion_dim

    def forward(self,x,his_bev,grid,normalize_factor,rt_flow,start_of_sequence):
        grid_ori=grid.unsqueeze(-2).repeat(1, 1,1,1,1,1)

        if self.offset_his is None or (~start_of_sequence).sum()==0:
            
            bin_prob_,bin_prob,samp_bias_,samp_bias,grid_=self.pred_bias(x,grid_ori,normalize_factor,rt_flow)
            if self.motion_his_flow_bin:
                self.offset_his=bin_prob_
            else:
                self.offset_his=samp_bias_
                if self.motion_his_learnable_lr:
                    self.offset_his=self.offset_his+self.offset_lr_net(x)[0][0][0][0].sum()*0.
            if self.motion_his_sup_w_his:
                self.offset_his_stack=[[]]*start_of_sequence.shape[0]
                self.offset_his_feat_stack=[[]]*start_of_sequence.shape[0]
                self.grid_stack=[[]]*start_of_sequence.shape[0]
                self.rt_flow_stack=[[]]*start_of_sequence.shape[0]
            self.iter=torch.zeros(start_of_sequence.shape[0],1,device=x.device)
        else:
            if sum(start_of_sequence)>0:
                self.iter[start_of_sequence]=0
                
            if self.motion_his_sup_w_his:
                for i in range(len(start_of_sequence)):
                    if start_of_sequence[i]:
                        self.offset_his_feat_stack[i]=[]
                        self.grid_stack[i]=[]
                for i in range(start_of_sequence.shape[0]):
        
                    for j in range(int(self.iter[i])):
                        if j==0:
                            
                            offset_his_feat=self.offset_his_feat_stack[i][j]
                            grid_i=self.grid_stack[i][j]
                            rt_flow_i=self.rt_flow_stack[i][j]
                            
                            bin_prob_i,bin_prob_i_softmax,samp_bias_i,samp_bias_i_norm,grid_i=self.pred_bias(offset_his_feat,grid_i,normalize_factor,rt_flow_i)
                            if self.motion_his_flow_bin:
                                self.offset_his[i:i+1]=bin_prob_i
                            else:
                                self.offset_his[i:i+1]=samp_bias_i
                        else:
                            
                            offset_his_feat=self.offset_his_feat_stack[i][j]
                            grid_i=self.grid_stack[i][j]
                            rt_flow_i=self.rt_flow_stack[i][j]
                            
                            with torch.enable_grad():
                                offset_his_i=self.offset_his[i:i+1]
                                
                                bin_prob_i,bin_prob_i_softmax,samp_bias_i,samp_bias_i_norm,_=self.pred_bias(offset_his_feat,grid_i,normalize_factor,rt_flow_i)
                                
                                samp_bias_i_=samp_bias_i.detach().clone().requires_grad_(True)
                                _,_,_,_,grid_i=self.pred_bias(None,grid_i,normalize_factor,rt_flow_i,samp_bias_i_)
                             
                                offset_his_grad_i=self.forward_his_(offset_his_i,grid_i,bin_prob_i,samp_bias_i_,normalize_factor,rt_flow_i)
                                offset_his_grad_i=offset_his_grad_i.clamp(-20,20)
                                
                                offset_his=self.offset_his.clone()
                                
                                valid_mask=((grid_i.permute(0,4,3,1,2)<1.)*(grid_i.permute(0,4,3,1,2)>-1.))
                                valid_mask=(valid_mask*valid_mask.flip([1]))>0
                                
                                if self.motion_his_flow_bin:
                                    
                                    bin_prob_i=bin_prob_i-offset_his_grad_i.reshape(offset_his_grad_i.shape[0],offset_his_grad_i.shape[1]//2,2,*offset_his_grad_i.shape[2:])
                                    offset_his[i:i+1]=bin_prob_i
                                else:
                                    if self.motion_his_learnable_lr:
                                        lr=self.offset_lr_net(offset_his_feat)
                                        lr=lr*self.motion_his_base_lr
                                    else:
                                        lr=self.motion_his_base_lr
                                    offset_his_grad_i=offset_his_grad_i*lr
                                    samp_bias_i_=samp_bias_i.clone()
                                    
                                    samp_bias_i_[valid_mask]=samp_bias_i_[valid_mask]-offset_his_grad_i[valid_mask]
                       
                                    offset_his[i:i+1]=samp_bias_i_
                                    
                                self.offset_his=offset_his
                                
            if sum(start_of_sequence)>0:
                bin_prob_,bin_prob,samp_bias_,samp_bias,grid_=self.pred_bias(x[start_of_sequence],grid_ori[start_of_sequence],normalize_factor,rt_flow[start_of_sequence]) 
                if self.motion_his_flow_bin:
                    self.offset_his[start_of_sequence]=bin_prob_
                else:
                    self.offset_his[start_of_sequence]=samp_bias_
        
            
            offset_his=self.offset_his[~start_of_sequence].clone()
            for i in range(offset_his.shape[0]):
                offset_his_non_start=[]
                with torch.enable_grad():
                    offset_his_i=offset_his[i:i+1]
                    
                    bin_prob_i,bin_prob_i_softmax,samp_bias_i,samp_bias_i_norm,_=self.pred_bias(x[~start_of_sequence][i:i+1],grid_ori[~start_of_sequence][i:i+1],normalize_factor,rt_flow[~start_of_sequence][i:i+1])
                
                    samp_bias_i_=samp_bias_i.detach().clone().requires_grad_(True)

                    _,_,_,_,grid_i=self.pred_bias(None,grid_ori[~start_of_sequence][i:i+1],normalize_factor,rt_flow[~start_of_sequence][i:i+1],samp_bias_i_)
                    
                        
                    offset_his_grad_i=self.forward_his_(offset_his_i,grid_i,bin_prob_i,samp_bias_i_,normalize_factor,rt_flow[~start_of_sequence][i:i+1])
                    
                    valid_mask=((grid_i[...,:self.motion_dim].permute(0,4,3,1,2)>1.)+(grid_i[...,:self.motion_dim].permute(0,4,3,1,2)<-1.))
                    valid_mask=((valid_mask.sum(1,keepdim=True))<1).expand(-1,self.motion_dim,-1,-1,-1)
                    
                    offset_his_grad_i=offset_his_grad_i.clamp(-20,20)
                    if self.motion_his_flow_bin:
                        
                        bin_prob_i=bin_prob_i-offset_his_grad_i.reshape(offset_his_grad_i.shape[0],offset_his_grad_i.shape[1]//2,2,*offset_his_grad_i.shape[2:])
                        offset_his_non_start.append(bin_prob_i)
                    else:
                        
                        if self.motion_his_learnable_lr:
                            lr=self.offset_lr_net(x[~start_of_sequence][i:i+1])
                            lr=lr*self.motion_his_base_lr
                        else:
                            lr=self.motion_his_base_lr
                        offset_his_grad_i=offset_his_grad_i*lr
                        samp_bias_i_=samp_bias_i.clone()
                        
                        samp_bias_i_[valid_mask]=samp_bias_i_[valid_mask]-offset_his_grad_i[valid_mask]
                        offset_his_non_start.append(samp_bias_i_)
                        
            offset_his_non_start=torch.stack(offset_his_non_start)
            self.offset_his[~start_of_sequence]=offset_his_non_start 
        if self.motion_his_flow_bin:
            bin_prob_his=self.offset_his.softmax(1)
            
            samp_bias=(bin_prob_his*self.offset_bin.to(bin_prob_his).reshape(1,-1,1,1,1,1)).sum(1)
        else:
            samp_bias=self.offset_his
        _,grid=self.fuse_grid_bias(grid_ori,samp_bias,normalize_factor,rt_flow)
        
        self.offset_his=self.offset_his.detach()
        self.iter+=1
        
        if self.motion_his_sup_w_his:
            for i in range(start_of_sequence.shape[0]):
                
                self.offset_his_feat_stack[i].append(x.detach()[i:i+1])
                self.grid_stack[i].append(grid_ori[i:i+1])
                self.rt_flow_stack[i].append(rt_flow[i:i+1])

        if self.bev_2d:
            sampled_history_bev = F.grid_sample(his_bev[:,:,0], grid.to(x.dtype)[...,0,:], align_corners=True, mode='bilinear')
            sampled_history_bev=sampled_history_bev.unsqueeze(2)
        else:
            sampled_history_bev = F.grid_sample(his_bev, grid.to(x.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode='bilinear')
        attn=torch.ones_like(x[:,:1,...]).to(x)
        attn=attn.unsqueeze(1)
        
        sampled_history_bev=sampled_history_bev.reshape(sampled_history_bev.shape[0],sampled_history_bev.shape[1],sampled_history_bev.shape[2]//attn.shape[2],attn.shape[2],*sampled_history_bev.shape[-2:])
        sampled_history_bev=sampled_history_bev.permute(0,1,3,2,4,5)
        sampled_history_bev=(sampled_history_bev*attn).sum(2)

        if self.bev_2d:
            sampled_history_bev=sampled_history_bev.squeeze(2)
        return sampled_history_bev,grid
    def forward_his(self,offset_his_i,grid_i,bin_prob,samp_bias,normalize_factor=None,rt_flow=None):
        if self.bev_2d:
            
            if self.motion_his_flow_bin:
                offset_his_i=offset_his_i.reshape(offset_his_i.shape[0],-1,*offset_his_i.shape[3:])
            
            sampled_history_offset = grid_sample_2d(offset_his_i[:,:,0], grid_i.to(offset_his_i.dtype)[...,0,:])
            # sampled_history_offset = F.grid_sample(offset_his_i[:,:,0], grid_[i:i+1].to(x.dtype)[...,0,:], align_corners=True, mode='bilinear')
            sampled_history_offset=sampled_history_offset.unsqueeze(2)
        else:
            sampled_history_offset = grid_sample_3d(offset_his_i,grid_i.to(offset_his_i.dtype).permute(0, 3, 1, 2, 4))
            # sampled_history_offset = F.grid_sample(offset_his_i, grid_[i:i+1].to(x.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode='bilinear')
        if self.motion_his_flow_bin:
            sampled_history_offset=sampled_history_offset.reshape(sampled_history_offset.shape[0],sampled_history_offset.shape[1]//2,2,*sampled_history_offset.shape[2:])
            inner_loss=-((sampled_history_offset.softmax(1)*bin_prob).sum(1)+(bin_prob.softmax(1)*sampled_history_offset).sum(1)).mean()/2
        else:
            inner_loss=(sampled_history_offset-samp_bias).pow(2).mean(1).sum()

        offset_his_grad_i=torch.autograd.grad(inner_loss,[samp_bias])[0]

        return offset_his_grad_i
    def forward_his_(self,offset_his_i,grid_i,bin_prob,samp_bias,normalize_factor,rt_flow):

        if self.bev_2d:
            
            sampled_history_offset,grid_grad=grid_sample_2d_grad_(offset_his_i[:,:,0], grid_i.to(offset_his_i.dtype)[...,0,:])
            sampled_history_offset=sampled_history_offset.unsqueeze(2)

            
            grid_grad=(grid_grad*2/normalize_factor[:2].reshape(1,-1,1,1,1))
            
            diff=2*(sampled_history_offset-samp_bias)
            offset_his_grad_i=torch.einsum('bcihw,bijhw->bcjhw',grid_grad,diff)-diff
            
            offset_his_grad_i=offset_his_grad_i/diff.shape[1]
        else:
            
            sampled_history_offset,grid_grad=grid_sample_3d_grad(offset_his_i, grid_i.to(offset_his_i.dtype))
            # sampled_history_offset=sampled_history_offset.unsqueeze(2)
            sampled_history_offset=sampled_history_offset.permute(0,1,4,2,3)
            grid_grad=grid_grad.permute(0,1,2,5,3,4)
            grid_grad=(grid_grad[:,:self.motion_dim]*2/normalize_factor[:self.motion_dim].reshape(1,-1,1,1,1,1))

            diff=2*(sampled_history_offset-samp_bias).unsqueeze(2)
            offset_his_grad_i=torch.einsum('bcizhw,bijzhw->bcjzhw',grid_grad,diff)

            offset_his_grad_i=offset_his_grad_i-diff
            
            offset_his_grad_i=offset_his_grad_i/diff.shape[1]
            offset_his_grad_i=offset_his_grad_i.squeeze(2)
        return offset_his_grad_i
    
    def pred_bias(self,x,grid,normalize_factor,rt_flow,samp_bias_=None):
        if samp_bias_ is None:
            if self.motion_his_flow_bin:
                bin_prob_=self.history_stream_offset_net(x)
                bin_prob_=bin_prob_.reshape(bin_prob_.shape[0],bin_prob_.shape[1]//2,2,*bin_prob_.shape[2:])
                bin_prob=bin_prob_.softmax(1)
                samp_bias_=(bin_prob*self.offset_bin.to(bin_prob).reshape(1,-1,1,1,1,1)).sum(1)
            else:
                
                samp_bias_=self.history_stream_offset_net(x)
                samp_bias_=samp_bias_.clamp(-20,20)
                
                bin_prob_=None
                bin_prob=None
        else:
            bin_prob_=None
            bin_prob=None
    
        samp_bias,grid_=self.fuse_grid_bias(grid,samp_bias_,normalize_factor,rt_flow)
        
        return bin_prob_,bin_prob,samp_bias_,samp_bias,grid_
    def fuse_grid_bias(self,grid,samp_bias,normalize_factor,rt_flow):
        samp_bias=samp_bias.permute(0,3,4,2,1)
        samp_bias=samp_bias.unsqueeze(-2)
        samp_bias=samp_bias+self.history_sample_bias[None,None,None,None,...].to(grid.device)
        samp_bias=samp_bias/normalize_factor[:self.motion_dim].view(1, 1, 1, 1, 1,self.motion_dim)*2
        grid_=grid.clone()
        grid_[...,:self.motion_dim]=grid_[...,:self.motion_dim]+samp_bias
        grid_=grid_.reshape(*grid_.shape[:-3],grid_.shape[-3]*grid_.shape[-2],grid_.shape[-1])
        return samp_bias,grid_

def grid_sample_2d(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def grid_sample_2d_grad(image, optical):
   
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)

    # Calculate corners
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    # Calculate interpolation weights
    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    # Clamping indices
    with torch.no_grad():
        ix_nw = torch.clamp(ix_nw, 0, IW-1)
        iy_nw = torch.clamp(iy_nw, 0, IH-1)
        ix_ne = torch.clamp(ix_ne, 0, IW-1)
        iy_ne = torch.clamp(iy_ne, 0, IH-1)
        ix_sw = torch.clamp(ix_sw, 0, IW-1)
        iy_sw = torch.clamp(iy_sw, 0, IH-1)
        ix_se = torch.clamp(ix_se, 0, IW-1)
        iy_se = torch.clamp(iy_se, 0, IH-1)

    # Flatten the image to facilitate the gather operation
    image = image.view(N, C, IH * IW)

    # Gather pixel values from image
    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    # Calculate output
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    # Calculate gradients with respect to ix and iy
    grad_output = torch.ones_like(out_val)  # Assume a simple case where gradient of loss is 1

    # Gradients for interpolation weights
    grad_nw = nw_val.view(N, C, H, W) * grad_output
    grad_ne = ne_val.view(N, C, H, W) * grad_output
    grad_sw = sw_val.view(N, C, H, W) * grad_output
    grad_se = se_val.view(N, C, H, W) * grad_output

    # Gradients with respect to ix and iy
    grad_ix = (
        (grad_nw * (-1) * (iy_se - iy)).sum(1) +
        (grad_ne * (1) * (iy_sw - iy)).sum(1) +
        (grad_sw * (-1) * (iy - iy_ne)).sum(1) +
        (grad_se * (1) * (iy - iy_nw)).sum(1)
    )

    grad_iy = (
        (grad_nw * (-1) * (ix_se - ix)).sum(1) +
        (grad_ne * (-1) * (ix - ix_sw)).sum(1) +
        (grad_sw * (1) * (ix_ne - ix)).sum(1) +
        (grad_se * (1) * (ix - ix_nw)).sum(1)
    )

    # Gradient wrt to optical grid (ix, iy are normalized to range [-1, 1])
    grad_optical_x = grad_ix * (IW - 1) / 2
    grad_optical_y = grad_iy * (IH - 1) / 2

    # Stack gradients together to get the final gradient for the optical flow grid
    grad_optical = torch.stack([grad_optical_x, grad_optical_y], dim=-1)

  
    return grad_optical

def grid_sample_2d_grad_(image, optical):
   
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)

    # Calculate corners
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    # Calculate interpolation weights
    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    # Clamping indices
    with torch.no_grad():
        ix_nw = torch.clamp(ix_nw, 0, IW-1)
        iy_nw = torch.clamp(iy_nw, 0, IH-1)
        ix_ne = torch.clamp(ix_ne, 0, IW-1)
        iy_ne = torch.clamp(iy_ne, 0, IH-1)
        ix_sw = torch.clamp(ix_sw, 0, IW-1)
        iy_sw = torch.clamp(iy_sw, 0, IH-1)
        ix_se = torch.clamp(ix_se, 0, IW-1)
        iy_se = torch.clamp(iy_se, 0, IH-1)

    # Flatten the image to facilitate the gather operation
    image = image.view(N, C, IH * IW)

    # Gather pixel values from image
    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    # Calculate output
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    # Calculate gradients with respect to ix and iy
    grad_output = torch.ones_like(out_val)  # Assume a simple case where gradient of loss is 1

    # Gradients for interpolation weights
    grad_nw = nw_val.view(N, C, H, W) * grad_output
    grad_ne = ne_val.view(N, C, H, W) * grad_output
    grad_sw = sw_val.view(N, C, H, W) * grad_output
    grad_se = se_val.view(N, C, H, W) * grad_output

    # Gradients with respect to ix and iy
    grad_ix = (
        (grad_nw * (-1) * (iy_se - iy)) +
        (grad_ne * (1) * (iy_sw - iy)) +
        (grad_sw * (-1) * (iy - iy_ne)) +
        (grad_se * (1) * (iy - iy_nw))
    )

    grad_iy = (
        (grad_nw * (-1) * (ix_se - ix)) +
        (grad_ne * (-1) * (ix - ix_sw)) +
        (grad_sw * (1) * (ix_ne - ix)) +
        (grad_se * (1) * (ix - ix_nw))
    )
    
    # Gradient wrt to optical grid (ix, iy are normalized to range [-1, 1])
    grad_optical_x = grad_ix * (IW - 1) / 2
    grad_optical_y = grad_iy * (IH - 1) / 2

    # Stack gradients together to get the final gradient for the optical flow grid
    grad_optical = torch.stack([grad_optical_x, grad_optical_y], dim=1)

    # grad_optical=grad_optical_x
  
    return out_val,grad_optical

def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)
    
    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().reshape(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().reshape(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().reshape(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().reshape(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().reshape(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().reshape(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().reshape(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().reshape(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val

def grid_sample_3d_grad(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    # Compute corner indices
    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)
        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw
        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw
        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw
        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1
        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1
        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1
        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    # Calculate interpolation weights
    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    # Clamp corner indices
    with torch.no_grad():
        ix_tnw = torch.clamp(ix_tnw, 0, IW - 1)
        iy_tnw = torch.clamp(iy_tnw, 0, IH - 1)
        iz_tnw = torch.clamp(iz_tnw, 0, ID - 1)
        ix_tne = torch.clamp(ix_tne, 0, IW - 1)
        iy_tne = torch.clamp(iy_tne, 0, IH - 1)
        iz_tne = torch.clamp(iz_tne, 0, ID - 1)
        ix_tsw = torch.clamp(ix_tsw, 0, IW - 1)
        iy_tsw = torch.clamp(iy_tsw, 0, IH - 1)
        iz_tsw = torch.clamp(iz_tsw, 0, ID - 1)
        ix_tse = torch.clamp(ix_tse, 0, IW - 1)
        iy_tse = torch.clamp(iy_tse, 0, IH - 1)
        iz_tse = torch.clamp(iz_tse, 0, ID - 1)
        ix_bnw = torch.clamp(ix_bnw, 0, IW - 1)
        iy_bnw = torch.clamp(iy_bnw, 0, IH - 1)
        iz_bnw = torch.clamp(iz_bnw, 0, ID - 1)
        ix_bne = torch.clamp(ix_bne, 0, IW - 1)
        iy_bne = torch.clamp(iy_bne, 0, IH - 1)
        iz_bne = torch.clamp(iz_bne, 0, ID - 1)
        ix_bsw = torch.clamp(ix_bsw, 0, IW - 1)
        iy_bsw = torch.clamp(iy_bsw, 0, IH - 1)
        iz_bsw = torch.clamp(iz_bsw, 0, ID - 1)
        ix_bse = torch.clamp(ix_bse, 0, IW - 1)
        iy_bse = torch.clamp(iy_bse, 0, IH - 1)
        iz_bse = torch.clamp(iz_bse, 0, ID - 1)

    # Flatten image for gather
    image = image.view(N, C, ID * IH * IW)

    # Gather values from corners
    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    # Calculate output
    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    # Gradient with respect to ix, iy, and iz
    # grad_output = torch.ones_like(out_val)  # Assuming a simple loss where dL/dout = 1
    
    # Compute gradients for interpolation weights with respect to ix, iy, iz
    grad_tnw = tnw_val.view(N, C, D, H, W)   # top-near-west corner
    grad_tne = tne_val.view(N, C, D, H, W)   # top-near-east corner
    grad_tsw = tsw_val.view(N, C, D, H, W)   # top-south-west corner
    grad_tse = tse_val.view(N, C, D, H, W)   # top-south-east corner
    grad_bnw = bnw_val.view(N, C, D, H, W)   # bottom-near-west corner
    grad_bne = bne_val.view(N, C, D, H, W)   # bottom-near-east corner
    grad_bsw = bsw_val.view(N, C, D, H, W)   # bottom-south-west corner
    grad_bse = bse_val.view(N, C, D, H, W)   # bottom-south-east corner

    # Compute partial derivatives for ix, iy, and iz
    
    ###########################
    grad_=torch.stack((grad_tnw,grad_tne,grad_tsw,grad_tse,grad_bnw,grad_bne,grad_bsw,grad_bse),dim=-1)
    grad_ix = torch.stack((
        (-1) * (iy_bse - iy) * (iz_bse - iz) ,
        (1) * (iy_bsw - iy) * (iz_bsw - iz) ,
        (-1) * (iy - iy_bne) * (iz_bne - iz) ,
         (1) * (iy - iy_bnw) * (iz_bnw - iz) ,
        (-1) * (iy_tse - iy) * (iz - iz_tse) ,
         (1) * (iy_tsw - iy) * (iz - iz_tsw) ,
        (-1) * (iy - iy_tne) * (iz - iz_tne) ,
        (1) * (iy - iy_tnw) * (iz - iz_tnw)
    ),dim=-1)
    grad_ix=torch.einsum('bwhzd,bnwhzd->bnwhz',grad_ix,grad_)
    
    grad_iy = torch.stack((
         (-1) * (ix_bse - ix) * (iz_bse - iz) ,
         (-1) * (ix - ix_bsw) * (iz_bsw - iz),
        (1) * (ix_bne - ix) * (iz_bne - iz) ,
         (1) * (ix - ix_bnw) * (iz_bnw - iz) ,
        (-1) * (ix_tse - ix) * (iz - iz_tse) ,
        (-1) * (ix - ix_tsw) * (iz - iz_tsw) ,
       (1) * (ix_tne - ix) * (iz - iz_tne) ,
        (1) * (ix - ix_tnw) * (iz - iz_tnw)
    ),dim=-1)
    grad_iy=torch.einsum('bwhzd,bnwhzd->bnwhz',grad_iy,grad_)
    
    grad_iz = torch.stack((
        (-1) * (ix_bse - ix) * (iy_bse - iy) ,
        (-1) * (ix - ix_bsw) * (iy_bsw - iy) ,
        (-1) * (ix_bne - ix) * (iy - iy_bne) ,
        (-1) * (ix - ix_bnw) * (iy - iy_bnw) ,
        (1) * (ix_tse - ix) * (iy_tse - iy) ,
        (1) * (ix - ix_tsw) * (iy_tsw - iy) ,
        (1) * (ix_tne - ix) * (iy - iy_tne) ,
        (1) * (ix - ix_tnw) * (iy - iy_tnw)
    ),dim=-1)
    grad_iz=torch.einsum('bwhzd,bnwhzd->bnwhz',grad_iz,grad_)
    
    

    # Convert gradients from index space to normalized space
    grad_optical_x = grad_ix * (IW - 1) / 2
    grad_optical_y = grad_iy * (IH - 1) / 2
    grad_optical_z = grad_iz * (ID - 1) / 2

    # Stack the gradients to get the final gradient for the optical grid
    
    grad_optical = torch.stack([grad_optical_x, grad_optical_y, grad_optical_z], dim=1)
    
    

    return out_val,grad_optical

class GeometryHistoryFusion(BaseModule):
    
                 
    def __init__(self,
                 feat_channel,
                 grid_config,
                 input_size=None,
                 downsample=16,
                 forward_post=False,
                 context_channel=256,
                 depth_his_sup_w_his=False,
                 per_pixel_weight=False,
               
               
                 gate_net_with_his=False,
                 
                 ):
        super().__init__()
        # Deal with geometry history
        
        self.feat_channel=feat_channel
        self.history_feat=None
        self.history_cat_num=1
        self.downsample=downsample
        self.frustum=self.create_frustum(grid_config['depth'],
                                                    input_size,
                                                    downsample=self.downsample)
        if not forward_post:
            self.first_emb=nn.Parameter(torch.randn(feat_channel,input_size[0]//downsample,input_size[1]//downsample)*0.02)
        if forward_post:
            if per_pixel_weight:
                feat_channel=1
                             
            if gate_net_with_his:
                self.gate_net=nn.Sequential(
                        nn.Conv2d(
                            self.feat_channel*2, self.feat_channel, kernel_size=1, stride=1, padding=0),
                        # nn.BatchNorm2d(context_channel//2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            self.feat_channel, feat_channel, kernel_size=1, stride=1, padding=0),
                        nn.Sigmoid()
                        )
            else:
                self.gate_net=nn.Sequential(
                        nn.Conv2d(
                            context_channel, context_channel//2, kernel_size=1, stride=1, padding=0),
                        # nn.BatchNorm2d(context_channel//2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            context_channel//2, feat_channel, kernel_size=1, stride=1, padding=0),
                        nn.Sigmoid()
                        )
                
        self.depth_his_sup_w_his=depth_his_sup_w_his
        self.per_pixel_weight=per_pixel_weight
    
        self.gate_net_with_his=gate_net_with_his
        
        
    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        # if self.sid:
        #     d_sid = torch.arange(self.D).float()
        #     depth_cfg_t = torch.tensor(depth_cfg).float()
        #     d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
        #                       torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
        #     d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)
        

        # D x H x W x 3
        return torch.stack((x, y, d), -1)
    
    def warp_prev(self,feat,metas,post_rots_prev,post_trans_prev,start_of_sequence):
        
        B,N,_,H,W=feat.shape
        D=self.feat_channel
        frustum =self.frustum.to(feat)
        
        points = frustum - metas['post_trans'][~start_of_sequence].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots'][~start_of_sequence]).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        rots = metas['k2s_sensor'][~start_of_sequence][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][~start_of_sequence][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins'][~start_of_sequence]))

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)
        neg_mask = points[..., 2, 0] < 1e-3
        points = metas['intrins'][~start_of_sequence].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points = torch.cat( (points[..., :2, :] / points[..., 2:3, :], points[..., 2:3, :]), 5)
        points = post_rots_prev[...,:3,:3].view(B, N, 1, 1, 1, 3, 3).matmul(
            points).squeeze(-1)

        points += post_trans_prev[...,:3].view(B, N, 1, 1, 1, 3)
        
        px = points[..., 0] / (W*self.downsample - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (H*self.downsample - 1.0) * 2.0 - 1.0
        pz = (points[..., 2]-self.frustum.reshape(-1,3).min(0)[0][-1]) / (self.frustum.reshape(-1,3).max(0)[0][-1]-self.frustum.reshape(-1,3).min(0)[0][-1]) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        pz[neg_mask] = -2
        grid = torch.stack([px, py,pz], dim=-1)
        grid = grid.view(B * N, D , H, W, 3)

        if len(feat.shape)==5:
            feat=feat.reshape(B*N,1,*feat.shape[2:])
        else:
            feat=feat.reshape(B*N,*feat.shape[2:])
        warp_prev = F.grid_sample(feat, grid,
                                      align_corners=True,
                                      padding_mode='zeros')
        warp_prev=warp_prev.view(B,N,D,H,W)
        return warp_prev
    def forward(self,feat,img_metas,stereo_metas, warp_his=True):
        if warp_his:
            seq_ids = torch.LongTensor([
                single_img_metas['sequence_group_idx'] 
                for single_img_metas in img_metas]).to(feat.device)
            start_of_sequence = torch.BoolTensor([
                single_img_metas['start_of_sequence'] 
                for single_img_metas in img_metas]).to(feat.device)

            if self.history_feat is None:
                self.history_seq_ids = seq_ids.clone()
            B,N,D,H,W=feat.shape
            if (~start_of_sequence).sum() > 0:
                feat_prev=self.history_feat[~start_of_sequence]
                if (start_of_sequence).sum() > 0:
                    warped_feat=self.first_emb.to(feat).unsqueeze(0).unsqueeze(0).repeat(B,N,1,1,1)
                    post_rots_prev=self.post_rots_prev[~start_of_sequence]
                    post_trans_prev=self.post_trans_prev[~start_of_sequence]
                    warped_feat[~start_of_sequence]=self.warp_prev(feat_prev,stereo_metas,post_rots_prev,post_trans_prev,start_of_sequence)
                    
                else:
                    warped_feat=self.warp_prev(feat_prev,stereo_metas,self.post_rots_prev,self.post_trans_prev,start_of_sequence)+self.first_emb[0][0][0]*0.
                
            else:
                warped_feat=self.first_emb.to(feat).unsqueeze(0).unsqueeze(0).repeat(B,N,1,1,1)
                
            
            self.post_rots_prev=stereo_metas['post_rots']
            self.post_trans_prev=stereo_metas['post_trans']
            
            assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
            self.history_seq_ids=seq_ids
            
            
            return warped_feat
            
        self.history_feat = feat.detach()
        
    def forward_post(self,context,feat,img_metas,stereo_metas,gt_depth=None):
        
        
            
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(feat.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(feat.device)
        
            
        if self.history_feat is None:
            # self.history_feat = feat.clone()
            self.history_seq_ids = seq_ids.clone()
            # self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            
            self.history_feat = feat.detach()
            
            # All 0s, representing current timestep.
            # self.history_sweep_time = feat.new_zeros(feat.shape[0], self.history_cat_num)
            if self.depth_his_sup_w_his:
                self.depth_his_stack=[[]]*start_of_sequence.shape[0]
                self.depth_his_feat_stack=[[]]*start_of_sequence.shape[0]
                self.post_rots_prev_stack=[[]]*start_of_sequence.shape[0]
                self.post_trans_prev_stack=[[]]*start_of_sequence.shape[0]
                self.stereo_metas_stack=[[]]*start_of_sequence.shape[0]
            self.iter=torch.zeros(start_of_sequence.shape[0],1,device=feat.device)
        
        B,N,D,H,W=feat.shape
        
        if sum(start_of_sequence)>0:
            self.iter[start_of_sequence]=0
        if self.depth_his_sup_w_his:
            if sum(start_of_sequence)>0:
                for i in range(len(start_of_sequence)):
                    if start_of_sequence[i]:
                        self.depth_his_feat_stack[i]=[]
                        self.depth_his_stack[i]=[]
                        self.post_rots_prev_stack[i]=[]
                        self.post_trans_prev_stack[i]=[]
                        self.stereo_metas_stack[i]=[]     
            for i in range(start_of_sequence.shape[0]):
                
                for j in range(int(self.iter[i])):
                    if j==0:
                        self.history_feat[i:i+1]=self.depth_his_stack[i][j]
                    else:
                        depth_his=self.depth_his_stack[i][j]
                        depth_his_feat=self.depth_his_feat_stack[i][j]
                        post_rots_prev=self.post_rots_prev_stack[i][j]
                        post_trans_prev=self.post_trans_prev_stack[i][j]
                        metas=self.stereo_metas_stack[i][j]
                        
                        fused_feat=depth_his.clone()
                        
                        feat_prev=self.history_feat[i:i+1]
                        start_of_sequence_=torch.Tensor([True]*B).bool()
                        start_of_sequence_[i]=False
                        warped_feat=self.warp_prev(feat_prev,metas,post_rots_prev,post_trans_prev,start_of_sequence_)
                        
                        valid_mask=warped_feat!=0
                        
                        depth_his_feat=depth_his_feat.reshape(N,-1,H,W)
                        if self.gate_net_with_his:
                            gate_in=torch.cat((warped_feat.flatten(0,1),depth_his.flatten(0,1)),dim=1)
                        else:
                            gate_in=depth_his_feat
                            
                        weight=self.gate_net(gate_in)
                        
                        weight=weight.reshape(1,N,*weight.shape[1:])
                        
                        feat_valid=(weight*warped_feat)+((1-weight)*depth_his)
                       
                        fused_feat[valid_mask]=feat_valid[valid_mask]
                        history_feat=self.history_feat.clone()
                        history_feat[i:i+1]=fused_feat
                        self.history_feat=history_feat

        if (~start_of_sequence).sum() > 0:
            feat_prev=self.history_feat[~start_of_sequence]
            fused_feat=feat.clone()
            if gt_depth is not None:
                
                fused_feat_gt=gt_depth.clone()
   
            post_rots_prev=self.post_rots_prev[~start_of_sequence]
            post_trans_prev=self.post_trans_prev[~start_of_sequence]
       
            warped_feat=self.warp_prev(feat_prev,stereo_metas,post_rots_prev,post_trans_prev,start_of_sequence)
            valid_mask=warped_feat!=0
            
            context_=context.reshape(B,N,*context.shape[1:])[~start_of_sequence]
            context_=context_.reshape((~start_of_sequence).sum()*N,*context_.shape[2:])
            
            
            if self.gate_net_with_his:
                gate_in=torch.cat((warped_feat.flatten(0,1),feat[~start_of_sequence].flatten(0,1)),dim=1)
            else:
                gate_in=context_
            
            weight=self.gate_net(gate_in)
            weight=weight.reshape((~start_of_sequence).sum(),N,*weight.shape[1:])
            
            feat_valid=(weight*warped_feat)+((1-weight)*feat[~start_of_sequence])
            fused_feat_non_start=fused_feat[~start_of_sequence].clone()
            fused_feat_non_start[valid_mask]=feat_valid[valid_mask]
            fused_feat[~start_of_sequence]=fused_feat_non_start
            
            ####################
            if gt_depth is not None:
                feat_valid=(weight*warped_feat)+((1-weight)*gt_depth[~start_of_sequence])
                fused_feat_gt_non_start=fused_feat_gt[~start_of_sequence].clone()
                fused_feat_gt_non_start[valid_mask]=feat_valid[valid_mask]
                fused_feat_gt[~start_of_sequence]=fused_feat_gt_non_start
                
        
        else:
            
            if self.gate_net_with_his:
                zz=self.gate_net(torch.cat((feat.flatten(0,1),feat.flatten(0,1)),dim=1))
            else:
                zz=self.gate_net(context)
            fused_feat=feat+zz[0][0][0][0].sum()*0.
           
            #####################################
            if gt_depth is not None:
                fused_feat_gt=gt_depth
        if self.depth_his_sup_w_his:
            
            for i in range(start_of_sequence.shape[0]):
                self.depth_his_stack[i].append(feat.detach()[i:i+1])
                if start_of_sequence[i]:
                    self.depth_his_feat_stack[i].append(None)
                    self.post_rots_prev_stack[i].append(None)
                    self.post_trans_prev_stack[i].append(None)
                    self.stereo_metas_stack[i].append(None)
                else:
                    context=context.reshape(B,N,-1,H,W)
                    self.depth_his_feat_stack[i].append(context.detach()[i:i+1])
                    self.post_rots_prev_stack[i].append(self.post_rots_prev[i:i+1])
                    self.post_trans_prev_stack[i].append(self.post_trans_prev[i:i+1])
                    self.stereo_metas_stack[i].append(stereo_metas)
            
        
        
        self.post_rots_prev=stereo_metas['post_rots']
        self.post_trans_prev=stereo_metas['post_trans']
        
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        self.history_seq_ids=seq_ids
        
        self.history_feat = fused_feat.detach()
        if gt_depth is not None:
            self.history_feat = fused_feat_gt.detach()
        
        self.iter+=1
          
        return fused_feat
