# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet3d.models.builder import NECKS
import torch.utils.checkpoint as cp

from mmdet3d.models.necks.soft_filling import broadcast_pred_linear_interpolation as soft_filling
from itertools import product


def gen_dx_bx(xbound, ybound, zbound):
    # import pdb;pdb.set_trace()
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@NECKS.register_module()
class LSSViewTransformerFunction(BaseModule):
    r"""Lift-Splat-Shoot view transformer.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        # in_channels=512,
        out_channels=64,
        accelerate=False,
        uniform=False,
        with_cp=False,
        num_classes=19,
        soft_filling=False,
        collapse_z=False,
        occ_2d=False,
        dataset='nuscenes',
        weight_flip=True,
        depth2occ_inter=False,
        torch_sparse_coor=False,
        depth_emb_dim=80,
        geometry_group=False,

    ):
        super(LSSViewTransformerFunction, self).__init__()
        self.uniform = uniform
        self.with_cp = with_cp
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['x'],
                               self.grid_config['y'],
                               self.grid_config['z'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.input_size = input_size
        self.frustum=self.create_frustum(grid_config['depth'], input_size, downsample)
        
        self.out_channels = out_channels
        self.accelerate = accelerate
        self.initial_flag = True
        self.num_classes=num_classes
        self.soft_filling=soft_filling
        self.grid_map=self.gen_grid_map(grid_size=self.grid_size)

        self.collapse_z=collapse_z
        self.occ_2d=occ_2d
        self.dataset=dataset
        self.weight_flip=weight_flip
        self.depth2occ_inter=depth2occ_inter
        
        self.torch_sparse_coor=torch_sparse_coor
        self.depth_emb_dim=depth_emb_dim
        self.geometry_group=geometry_group
    def gen_grid_map(self,grid_size):
        w,h,z=grid_size.long().tolist()  
       
        xs = torch.linspace(0, h - 1, h).view(h, 1, 1).expand(h, w, z)
        ys = torch.linspace(0, w - 1,w).view(1, w, 1).expand(h, w, z)
        zs = torch.linspace(0, z- 1, z).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs), -1).view(1, h, w, z, 3)+0.5

        
        return grid    
    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

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
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        frustum = torch.stack((x, y, d), -1)
        return frustum

    
    def img_to_ego_coor(self,coor,rots, trans, cam2imgs, post_rots, post_trans,
                       bda):

        B, N, _ = trans.shape
        points = coor.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points
    def ego_to_img_coor(self,coor,rots, trans, cam2imgs, post_rots, post_trans,
                       bda):
        B, N, _ = trans.shape
        combine = rots.matmul(torch.inverse(cam2imgs))
        coor=torch.inverse(bda).view(B, 1, 1, 1, 1, 3,3).matmul(coor.unsqueeze(-1)).squeeze(-1)
        coor=coor-trans.view(B, N, 1, 1, 1, 3)
        coor=torch.inverse(combine).view(B, N, 1, 1, 1, 3, 3).matmul(coor.unsqueeze(-1))
        coor[...,2,:]=torch.where(coor[...,2,:]<=0,torch.ones_like(coor[...,2,:])*1e-6,coor[...,2,:])
        coor=torch.cat((coor[..., :2,:]/coor[..., 2:3,:], coor[..., 2:3,:]), 5)
        coor=post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(coor)
        coor=coor.squeeze(-1)+post_trans.view(B, N, 1, 1, 1, 3)
        return coor
    
    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans,
                       bda,coor=None):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _ = trans.shape

        # post-transformation
        # B x N x D x H x W x 3
        if coor is None:
            points = self.frustum.to(rots)
        else:
            points = coor

        points = points - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, bda,coor=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        if coor is None:
            points = self.frustum.to(rots)
        else:
            points = coor
        # undo post-transformation
        # B x N x D x H x W x 3
        points =points - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        
        if intrins.shape[3] == 4: # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def init_acceleration_v2(self, coor,grid_size=None):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """
        if grid_size==None:
            grid_size=self.grid_size
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor,grid_size=grid_size)
        
        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, geometry, feat,grid_size=None,**kwargs):
        if self.torch_sparse_coor:
            B, N, D, H, W, _ = coor.shape
            num_points = B * N * D * H * W
            coor_ = coor.long().view(num_points, 3)
            batch_idx = torch.arange(0, B ).reshape(B, 1). \
                expand(B, num_points // B).reshape(num_points, 1).to(coor_)
            coor_ = torch.cat((coor_, batch_idx), 1)

            # filter out points that are outside box
            kept = (coor_[:, 0] >= 0) & (coor_[:, 0] < grid_size[0]) & \
                (coor_[:, 1] >= 0) & (coor_[:, 1] < grid_size[1]) & \
                (coor_[:, 2] >= 0) & (coor_[:, 2] < grid_size[2])
            if len(kept) == 0:
                return None, None, None, None, None
            
            coor_= coor_[kept]

            depth_=geometry.unsqueeze(3)
            feat_=feat.unsqueeze(2)
            weighted_feat=(depth_*feat_).permute(0,1,2,4,5,3).reshape(num_points,-1)
            
            weighted_feat =weighted_feat[kept]
        
            coor_=torch.flip(coor_,[-1])
            bev_feat_shape = (geometry.shape[0], int(grid_size[2]),
                            int(grid_size[1]), int(grid_size[0]),
                            weighted_feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat=torch.sparse_coo_tensor(coor_.t(),weighted_feat,bev_feat_shape).to_dense()
            
            bev_feat= bev_feat.permute(0, 4,  2, 3,1).contiguous()

        else:
            ranks_bev, ranks_depth, ranks_feat, \
                interval_starts, interval_lengths = \
                self.voxel_pooling_prepare_v2(coor,grid_size)
            if ranks_feat is None:
                print('warning ---> no points within the predefined '
                    'bev receptive field')
                dummy = torch.zeros(size=[
                    feat.shape[0], feat.shape[2],
                    int(grid_size[0]),
                    int(grid_size[1]),
                    int(grid_size[2]),
                ]).to(feat)
                # dummy = torch.cat(dummy.unbind(dim=2), 1)
                return dummy
            feat = feat.permute(0, 1, 3, 4, 2)
            bev_feat_shape = (geometry.shape[0], int(grid_size[2]),
                            int(grid_size[1]), int(grid_size[0]),
                            feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(geometry, feat, ranks_depth, ranks_feat, ranks_bev,
                                bev_feat_shape, interval_starts,
                                interval_lengths)
            bev_feat = bev_feat.permute(0, 1, 3, 4, 2) # B, C, Z, X, Y- > B, C, X, Y, Z
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor,grid_size=None):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(
            0, num_points // D , dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B ).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
       
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            grid_size[2] * grid_size[1] * grid_size[0])
        ranks_bev += coor[:, 2] * (grid_size[1] * grid_size[0])
        ranks_bev += coor[:, 1] * grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, cam_params):
        if self.initial_flag:
            
            coor = self.get_lidar_coor(*cam_params)
            coor = ((coor - self.grid_lower_bound.to(coor)) /
                        self.grid_interval.to(coor))
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def soft_filling_func(self,coor):
        B,N,K,H,W,_=coor.shape
        coor=coor.reshape(-1,3)
        weight,coor=soft_filling(coor,weight_flip=self.weight_flip)
        coor=coor.reshape(B,N,K,H,W,8,3)
        coor=coor.permute(0,1,2,5,3,4,6)
        coor=coor.reshape(B,N,coor.shape[2]*coor.shape[3],*coor.shape[4:]).contiguous()
        return weight,coor
    
    def view_transform_core(self, cam_params, geometry, tran_feat,coor=None,grid_size=None,coor_offsets=None,**kwargs):
        if self.dataset=='kitti':
            self.get_lidar_coor=self.get_geometry

        if self.accelerate:
            feat = tran_feat # tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            geometry = geometry #.view(B, N, self.D, H, W)
            bev_feat_shape = (geometry.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(geometry, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)
            bev_feat=bev_feat.permute(0,1,3,4,2)
    
        else:
            if coor_offsets is not None or self.depth2occ_inter:
                coor = self.frustum
                if coor_offsets is not None:
                    b,n,d,h,w=geometry.shape
                    coor_offsets=coor_offsets.reshape(-1,n,d,3,h,w).permute(0,1,2,4,5,3)
                    coor = coor.to(geometry)[None,None,...]+coor_offsets
                else:
                    coor = coor.to(geometry)[None,None,...].repeat(geometry.shape[0],geometry.shape[1],1,1,1,1)

                if self.depth2occ_inter:
                    depth2occ_inter_weight,depth2occ_inter_bias=kwargs['depth_output']['depth2occ_inter_weight'],kwargs['depth_output']['depth2occ_inter_bias']
                    max_idx=geometry.max(2,keepdim=True)[1]
                    hidden_idx=max_idx+torch.tensor([-1,0,1]).unsqueeze(-1).unsqueeze(-1).to(geometry)
                    hidden_idx=hidden_idx.long()
                    
                    mask=((hidden_idx>geometry.shape[1]-1) + hidden_idx<0)>0
                    hidden_idx=hidden_idx.clamp(0,geometry.shape[1]-1)
                    
                    
                    hidden_coor=torch.gather(coor,2,hidden_idx.unsqueeze(-1).repeat(1,1,1,1,1,3)).contiguous()
                    
                    hidden_depth=torch.gather(geometry,2,hidden_idx).contiguous()
                    # import pdb;pdb.set_trace()
                    b,n,num_hidden,h,w=hidden_depth.shape
                    if self.geometry_group:
                        depth2occ_inter_weight=depth2occ_inter_weight.reshape(b//self.geometry_group,n,*depth2occ_inter_weight.shape[1:])
                        depth2occ_inter_weight=depth2occ_inter_weight.unsqueeze(1)
                        depth2occ_inter_weight=depth2occ_inter_weight.expand(-1,self.geometry_group,-1,-1,-1,-1,-1,-1)
                        depth2occ_inter_weight=depth2occ_inter_weight.reshape(depth2occ_inter_weight.shape[0]*self.geometry_group,*depth2occ_inter_weight.shape[2:])

                        depth2occ_inter_bias=depth2occ_inter_bias.reshape(b//self.geometry_group,n,*depth2occ_inter_bias.shape[1:])
                        depth2occ_inter_bias=depth2occ_inter_bias.unsqueeze(1)
                        depth2occ_inter_bias=depth2occ_inter_bias.expand(-1,self.geometry_group,-1,-1,-1,-1,-1,-1)
                        depth2occ_inter_bias=depth2occ_inter_bias.reshape(depth2occ_inter_bias.shape[0]*self.geometry_group,*depth2occ_inter_bias.shape[2:])
                    else:
                        depth2occ_inter_weight=depth2occ_inter_weight.reshape(b,n,*depth2occ_inter_weight.shape[1:])
                        depth2occ_inter_bias=depth2occ_inter_bias.reshape(b,n,*depth2occ_inter_bias.shape[1:])
                        
                    hidden_depth=hidden_depth.unsqueeze(2).unsqueeze(2)*depth2occ_inter_weight
                    
                    
                    depth2occ_inter_bias=depth2occ_inter_bias.permute(0,1,2,3,5,6,4).unsqueeze(4)
                    
                    hidden_coor=hidden_coor.unsqueeze(2).unsqueeze(2).repeat(1,1,depth2occ_inter_bias.shape[2],depth2occ_inter_bias.shape[3],1,1,1,1)
                    
                    hidden_coor[:,:,hidden_coor.shape[2]//2,hidden_coor.shape[3]//2,...]=10000.
                    hidden_coor[...,:2]=depth2occ_inter_bias+hidden_coor[...,:2]
                    hidden_coor=hidden_coor.permute(0,1,4,5,6,7,2,3)
                    hidden_coor[mask]=10000.
                    hidden_coor=hidden_coor.permute(0,1,6,7,2,3,4,5)
                    
                    hidden_coor=hidden_coor.reshape(b,n,hidden_coor.shape[2]*hidden_coor.shape[3]*hidden_coor.shape[4],h,w,3)
                    hidden_depth=hidden_depth.reshape(*hidden_coor.shape[:-1])
                    
                    geometry=torch.cat((geometry,hidden_depth),2)
                    coor=torch.cat((coor,hidden_coor),2)
                            
                
                coor = self.get_lidar_coor(*cam_params,coor)
            else:
                coor = self.get_lidar_coor(*cam_params)
            # convert coordinate into the voxel space
            
            coor = ((coor - self.grid_lower_bound.to(coor)) /
                    self.grid_interval.to(coor))
            if self.geometry_group:
                coor=coor.unsqueeze(1)
                coor=coor.expand(-1,self.geometry_group,-1,-1,-1,-1,-1)
                coor=coor.reshape(coor.shape[0]*self.geometry_group,coor.shape[2],*coor.shape[3:])
            if self.soft_filling:
                B,N,D,H,W=geometry.shape

                weight,coor=self.soft_filling_func(coor)
                weight=weight.reshape(B,N,D,H,W,8)
                geometry=geometry.unsqueeze(-1)*weight
                geometry=geometry.permute(0,1,2,5,3,4)
                geometry=geometry.reshape(B,N,geometry.shape[2]*geometry.shape[3],*geometry.shape[4:]).contiguous()

            if grid_size==None:
                grid_size=self.grid_size
            bev_feat = self.voxel_pooling_v2(
                coor, geometry,
                tran_feat,grid_size,**kwargs)

        return bev_feat
    

    def view_transform(self, cam_params, geometry, tran_feat,coor_offsets=None, **kwargs):
        if self.accelerate:
            self.pre_compute(cam_params)
        if self.geometry_group:
            geometry=geometry.reshape(geometry.shape[0],geometry.shape[1]//self.geometry_group,self.geometry_group,*geometry.shape[2:])
            geometry=geometry.permute(0,2,1,3,4,5)
            geometry=geometry.reshape(geometry.shape[0]*self.geometry_group,*geometry.shape[2:])
            tran_feat=tran_feat.reshape(tran_feat.shape[0],tran_feat.shape[1],self.geometry_group,-1,*tran_feat.shape[3:])
            tran_feat=tran_feat.permute(0,2,1,3,4,5)
            tran_feat=tran_feat.reshape(tran_feat.shape[0]*self.geometry_group,*tran_feat.shape[2:])
        
        bev_feat=self.view_transform_core(cam_params, geometry, tran_feat,coor_offsets=coor_offsets,**kwargs)
            
        if self.geometry_group:
            bev_feat=bev_feat.reshape(cam_params[0].shape[0],-1,*bev_feat.shape[2:])
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        if self.occ_2d:
            bev_feat=bev_feat.sum(-1)

        return bev_feat

    # @run_time('lss3d')
    def forward(self, cam_params, context, geometry,coor_offsets=None, **kwargs):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """

        bev = self.view_transform(cam_params, geometry, context,coor_offsets=coor_offsets, **kwargs)
        return bev