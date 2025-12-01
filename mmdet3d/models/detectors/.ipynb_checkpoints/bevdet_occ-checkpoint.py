# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
import os.path as osp
from mmcv.runner import auto_fp16

def compute_errors(pred,gt):
    # print('gt',gt,'pred',pred,pred.min(),pred.max())
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)
class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg
class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 wo_depth_sup=False,
                 sup_adaptive_depth=False,
                 adaptive_depth_bin=False,
                 disc2continue_depth_continue_sup=False,
                 lift_attn =False,
                 supervise_intermedia=False,
                 sup_binary=False,
                 only_sup_sem=False,
                 lift_attn_wo_depth_sup=False,
                 depth2occ=False,
                 use_gt_occ2depth=False,
                 inter_sup_non_mask=False,
                 add_occ_depth_loss=False,
                 binary_non_mask=False,
                 fuse_his_attn=False,
                 fuse_self=False,
                #  only_train_depth=False,
                 dispart_loss=False,
                 fuse_self_round=1,
                 lift_attn_with_ori_feat_add=False,
                 final_binary_non_mask=False,
                 wo_stereo=False,
                 fuse_not_detach=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        if not self.only_train_depth:
            self.final_conv = ConvModule(
                            out_dim,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            conv_cfg=dict(type='Conv3d'))
            self.use_predicter =use_predicter
            if use_predicter:
                self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim*2, num_classes),
                )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.wo_depth_sup = wo_depth_sup
        self.sup_adaptive_depth=sup_adaptive_depth
        self.adaptive_depth_bin=adaptive_depth_bin
        self.depth_metrics = RunningAverageDict()
        self.disc2continue_depth_continue_sup=disc2continue_depth_continue_sup
        self.lift_attn =lift_attn 
        self.supervise_intermedia=supervise_intermedia
        self.sup_binary=sup_binary
        self.only_sup_sem=only_sup_sem
        self.lift_attn_wo_depth_sup = lift_attn_wo_depth_sup
        self.depth2occ=depth2occ
        self.use_gt_occ2depth=use_gt_occ2depth
        self.inter_sup_non_mask=inter_sup_non_mask
        self.add_occ_depth_loss=add_occ_depth_loss
        self.fuse_his_attn=fuse_his_attn
        self.binary_non_mask=binary_non_mask
        self.fuse_self=fuse_self
        self.dispart_loss=dispart_loss
        self.fuse_self_round=fuse_self_round
        self.lift_attn_with_ori_feat_add=lift_attn_with_ori_feat_add
        self.final_binary_non_mask=final_binary_non_mask
        self.wo_stereo=wo_stereo
        self.fuse_not_detach=fuse_not_detach
        # import pdb;pdb.set_trace()
        if self.wo_stereo:
            self.num_frame-=self.extra_ref_frames
            self.extra_ref_frames=0
        # self.only_train_depth=only_train_depth

    def loss_single(self,voxel_semantics,mask_camera,preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_
    def loss_binary(self,inputs,targets,mask):
        # import pdb;pdb.set_trace()
        loss = dict()
        mask=mask.bool()
        inputs=inputs[mask]
        # sem=inputs[:,:-1]
        occ=inputs[...,-1]
        
        targets=targets[mask]

        targets_occ=targets!=17
        
        # mask_occ=(targets_occ).nonzero().squeeze()
        # mask_free=(targets==17).nonzero().squeeze()
        # sem=sem[mask_occ]
        
        loss_occ=F.binary_cross_entropy_with_logits(occ,targets_occ.float())
        # sem=(sem+1e-6).log()
        # loss_sem=F.cross_entropy(sem,targets[mask_occ])
        loss['loss_occ'] = loss_occ
        # loss['loss_sem'] = loss_sem
        # loss=loss_occ+loss_sem
        # print((sem.sum(1)==0).sum(),2222222222222222222,loss_occ,loss_sem)
        return loss
    def TwoPart_loss(self,inputs,targets,mask=None,occ_depth=None,occ_depth_gt=None):
        # import pdb;pdb.set_trace()
        loss = dict()
        if self.add_occ_depth_loss:
            # import pdb;pdb.set_trace()
            depth_loss_occ=self.img_view_transformer.get_depth_loss_gt_occ2depth( occ_depth+1e-8,occ_depth_gt.permute(0,2,3,1).reshape(-1,occ_depth_gt.shape[1]))
            # depth_loss_occ=-((occ_depth+1e-8).log()*occ_depth_gt).sum(1).mean()
            loss['depth_loss_occ']=depth_loss_occ
        if mask is not None:
            mask=mask.bool()
            inputs_=inputs[mask]
            targets_=targets[mask]
        else:
            inputs_=inputs.reshape(-1,inputs.shape[-1])
            targets_=targets.reshape(-1)
        sem=inputs_[:,:-1]
        occ=inputs_[:,-1]
        
        

        targets_occ=targets_!=17
        
        
        # mask_free=(targets==17).nonzero().squeeze()
        
            
        
        if not self.only_sup_sem:
            if self.binary_non_mask:
                targets=targets.reshape(-1)
                targets=targets!=17
                occ=inputs.reshape(-1,inputs.shape[-1])[:,-1]
                loss_occ=F.binary_cross_entropy_with_logits(occ,targets.float())
            else:
                loss_occ=F.binary_cross_entropy_with_logits(occ,targets_occ.float())
            # sem=(sem+1e-6).log()
            
            loss['loss_occ'] = loss_occ
        if not self.sup_binary:
            mask_occ=(targets_occ).nonzero().squeeze()
            sem=sem[mask_occ]
            loss_sem=F.cross_entropy(sem,targets_[mask_occ])
            loss['loss_sem'] = loss_sem
        # loss=loss_occ+loss_sem
        # print((sem.sum(1)==0).sum(),2222222222222222222,loss_occ,loss_sem)
        return loss
    
    def TwoPart_loss_main(self,inputs,targets,mask=None,occ_depth=None,occ_depth_gt=None):
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
            targets_=targets[mask]
        else:
            inputs_=inputs.reshape(-1,inputs.shape[-1])
            targets_=targets.reshape(-1)
        sem=inputs_[:,:-1]
        occ=inputs_[:,-1]
        
        

        targets_occ=targets_!=17
        
        
        # mask_free=(targets==17).nonzero().squeeze()
        
            
        
        # if not self.only_sup_sem:
        if self.final_binary_non_mask:
            targets=targets.reshape(-1)
            targets=targets!=17
            occ=inputs.reshape(-1,inputs.shape[-1])[:,-1]
            loss_occ=F.binary_cross_entropy_with_logits(occ,targets.float())
        else:
            loss_occ=F.binary_cross_entropy_with_logits(occ,targets_occ.float())
            # sem=(sem+1e-6).log()
            
        loss['loss_occ'] = loss_occ
        # if not self.sup_binary:
        
        
        mask_occ=(targets_occ).nonzero().squeeze()
        sem=sem[mask_occ]
        loss_sem=F.cross_entropy(sem,targets_[mask_occ])
        loss['loss_sem'] = loss_sem
        # loss=loss_occ+loss_sem
        # print((sem.sum(1)==0).sum(),2222222222222222222,loss_occ,loss_sem)
        return loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        
        # if return_loss:
        #     return self.forward_test(**kwargs)
            
        # else:
        #     return self.forward_train(**kwargs)########################################
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    # vis_depth=False,
                    **kwargs):
        """Test function without augmentaiton."""

        # import pdb;pdb.set_trace()
        self.img_view_transformer.do_history=True
        img_feats, _, depth = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        if 'vis_depth' in kwargs:
            if kwargs['vis_depth']:
                return depth

        # import pdb;pdb.set_trace()
        if self.fuse_self:
            if self.lift_attn_with_ori_feat_add:
                interoccs=[]
                # import pdb;pdb.set_trace()
                for round in range(self.fuse_self_round):
                    depth,fuse_args=depth

                    # inter_occ1=depth[1]
                    x,bev_feat_list=img_feats
                    
                    img_feats, _, depth = self.extract_feat(
                        points, img=img, img_metas=img_metas,fuse_self_args=[fuse_args,[x.detach(),bev_feat_list],round+1], **kwargs)
                    
                    depth_,fuse_args=depth
                    inter_occ=depth_[1]
                    # import pdb;pdb.set_trace()
                    interoccs.extend(inter_occ)
                depth=[depth[0][0],interoccs]
                # import pdb;pdb.set_trace()
            else:
                depth,fuse_args=depth
                img_feats, _, _ = self.extract_feat(
                    points, img=img, img_metas=img_metas,fuse_self_args=[fuse_args,[img_feats[0].detach(),img_feats[1]]], **kwargs)
    
        if self.fuse_his_attn:
            depth,bda=depth
            self.img_view_transformer.fuse_history(img_feats[0],img_metas,bda,update_history=True)
        ####################evaluate depth####################
        if self.only_train_depth:
            gt_depth = kwargs['gt_depth'][0]
            loss_depth,depth,gt_depth= self.img_view_transformer.get_continue_gt_depth_and_pred_depth(gt_depth, depth)
            self.depth_metrics.update(compute_errors(depth.cpu().numpy(),gt_depth.cpu().numpy()))
            return [img_feats]
    ################################
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        ##########
          ##############
        # # occ_pred = occ_pred[0]
        # gt_occ=kwargs['voxel_semantics'][0]
        # mask_camera=kwargs['mask_camera'][0].bool()
        # occ_pred_mask=occ_pred[mask_camera]
        # gt_occ_mask=gt_occ[mask_camera]
        # loss=F.cross_entropy(occ_pred_mask,gt_occ_mask)
        # import pdb;pdb.set_trace()
        ###########3
        ###########
        if self.dispart_loss:
            pred_sem = occ_pred[..., :-1]
            pred_occ = occ_pred[..., -1:].sigmoid()
            pred_sem_category = pred_sem.argmax(-1)
            pred_free_category = (pred_occ<0.5).squeeze(-1)
            # import pdb; pdb.set_trace()
            pred_sem_category[pred_free_category] = 17
            occ_res = pred_sem_category
        else:    
            occ_score=occ_pred.softmax(-1)
            if 'vis_class' in kwargs:
                ##########
                mask=kwargs['voxel_semantics'][0]==kwargs['vis_class']
                mask=mask*kwargs['mask_camera'][0]
                occ_score=occ_score[mask].mean(0)
                # import pdb;pdb.set_trace()
                # occ_score=occ_score.mean((0,1,2,3))
                
                return occ_score
            ###############

            # import pdb;pdb.set_trace()
            occ_res=occ_score.argmax(-1)
        
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        ########
        # occ_gt2 = np.load(os.path.join('data/nuscenes/gts',img_metas[0]['scene_name'],img_metas[0]['sample_idx'],'labels.npz'))['semantics'][None]
        
        # print((torch.Tensor(occ_gt2).to(gt_occ)-gt_occ).sum(),img_metas[0]['sample_idx'],222)
                
        # import pdb;pdb.set_trace()
        # save_test_dir='data/nuscenes/gts_test_zz'
        # if save_test_dir is not None:
        #     if not osp.exists(save_test_dir):
        #         os.makedirs(save_test_dir,exist_ok=True)
        #     for i in range(len(img_metas)):
        #         save_path=osp.join(save_test_dir,img_metas[i]['sample_idx'])+'.npy'
        #         np.save(save_path,gt_occ.cpu().detach().numpy())
        #         # np.save(save_path,occ_pred[i].cpu().detach().numpy())
        #         print('save occ_pred to {}'.format(save_path))
        ##########
        result_dict = dict()
        result_dict['pred_occupancy'] = occ_res
        result_dict['index'] = img_metas[0]['index']
        return [result_dict]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        #################
        # import pdb;pdb.set_trace()
        # img_inputs=[img_inputs[0][i] for i in range(len(img_inputs[0]))]
        # img_metas=[img_metas[0]]
        ##################3
        
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        # import pdb;pdb.set_trace()
        # for i in range(len(img_metas)):
        #     print(img_metas[i]['scene_name'],img_metas[i]['sample_idx'],img_metas[i]['index'],img_feats[0].device)
          
            
            
            #depth[1]=inter_occ1+depth[1]
            # img_feats=img_feats[0]
        if self.fuse_self:
            if self.lift_attn_with_ori_feat_add:
                interoccs=[]
                # import pdb;pdb.set_trace()
                for round in range(self.fuse_self_round):
                    depth,fuse_args=depth

                    # inter_occ1=depth[1]
                    x,bev_feat_list=img_feats
                    # import pdb;pdb.set_trace()
                    #####################
                    if self.fuse_not_detach:
                        zz=x.clone()
                        zz=zz.permute(0,4,3,2,1)
                        zz[~kwargs['mask_camera']]=zz[~kwargs['mask_camera']].detach()
                        x=zz.permute(0,4,3,2,1)
                    else:
                    #################
                        x=x.detach()
                    #######################
                    img_feats, _, depth = self.extract_feat(
                        points, img=img_inputs, img_metas=img_metas,fuse_self_args=[fuse_args,[x,bev_feat_list],round+1], **kwargs)
                    
                    depth_,fuse_args=depth
                    inter_occ=depth_[1]
                    # import pdb;pdb.set_trace()
                    interoccs.extend(inter_occ)
                depth=[depth[0][0],interoccs]
                # import pdb;pdb.set_trace()
            else:
                depth,fuse_args=depth

                inter_occ1=depth[1]
                img_feats, _, depth = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas,fuse_self_args=[fuse_args,[img_feats[0].detach(),img_feats[1]]], **kwargs)
                
                #depth[1]=inter_occ1+depth[1]
                # img_feats=img_feats[0]
        
        if self.fuse_his_attn:
            depth,bda=depth
            self.img_view_transformer.fuse_history(img_feats[0],img_metas,bda,update_history=True)
        if self.supervise_intermedia:
            if self.add_occ_depth_loss:
                depth,inter_occs,weight,weight_gt=depth
            
            
            else:
                depth,inter_occs=depth
                weight=None
                weight_gt=None
            
        gt_depth = kwargs['gt_depth']
        losses = dict()
        if self.lift_attn_wo_depth_sup:
            depth=depth[0]
        if self.depth2occ and not self.lift_attn:
            depth,occ,occ_weight_=depth
            
        if not self.wo_depth_sup:
            if self.sup_adaptive_depth or self.only_train_depth or self.disc2continue_depth_continue_sup:
                # import pdb;pdb.set_trace()
                if self.lift_attn and not self.lift_attn_wo_depth_sup:
                    loss_depths=[]
                    for i in range(len(depth)):
                        loss_depths.append(self.img_view_transformer.get_continue_depth_loss(gt_depth, depth[i]))
                else:
                    loss_depth=self.img_view_transformer.get_continue_depth_loss(gt_depth, depth)
            elif self.use_gt_occ2depth:
                depth,gt_depth=depth
                loss_depth = self.img_view_transformer.get_depth_loss_gt_occ2depth( depth,gt_depth)
            else:
                if self.lift_attn and not self.lift_attn_wo_depth_sup:
                    loss_depths=[]
                    for i in range(len(depth)):
                        loss_depths.append(self.img_view_transformer.get_depth_loss(gt_depth, depth[i]))
                else:
                    loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            if self.lift_attn and not self.lift_attn_wo_depth_sup:
                for i in range(len(loss_depths)):
                    losses['loss_depth'+str(i)] = loss_depths[i]
            else:
                losses['loss_depth'] = loss_depth
        if self.only_train_depth:
            return losses
        
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
       
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.dispart_loss:
            # import pdb;pdb.set_trace()
            loss_occ = self.TwoPart_loss_main(occ_pred,voxel_semantics, mask_camera)
        else:   
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
         ##############
        # gt_occ=kwargs['voxel_semantics']
        # mask_camera=kwargs['mask_camera'].bool()
        # occ_pred_mask=occ_pred[mask_camera]
        # gt_occ_mask=gt_occ[mask_camera]
        # loss=F.cross_entropy(occ_pred_mask,gt_occ_mask)
        # import pdb;pdb.set_trace()
        # # for i in range(len(img_metas)):
        # #     if os.path.join(img_metas[i]['sample_idx']+'.npy') in os.listdir('data/nuscenes/gts_test_zz'):
        # #         occ_gt_ = np.load(os.path.join('data/nuscenes/gts_test_zz',img_metas[i]['sample_idx']+'.npy'))
        # #         occ_gt2 = np.load(os.path.join('data/nuscenes/gts',img_metas[i]['scene_name'],img_metas[i]['sample_idx'],'labels.npz'))['semantics'][None]
        # #         print((torch.Tensor(occ_gt_).to(gt_occ)-gt_occ[i:i+1]).sum(),img_metas[i]['sample_idx'],111)
        # #         print((torch.Tensor(occ_gt2).to(gt_occ)-gt_occ[i:i+1]).sum(),img_metas[i]['sample_idx'],222)
        ###########3
        losses.update(loss_occ)
        if self.supervise_intermedia:
            # if self.sup_binary:
            #     for i in range(len(inter_occs)):
            #         loss_occ=self.loss_binary(inter_occs[i],voxel_semantics, mask_camera)
            #         loss_occ = {key+str(i):value for key,value in loss_occ.items()}
            #         losses.update(loss_occ)
            # else:
            for i in range(len(inter_occs)):
                # loss_occ = self.loss_single(voxel_semantics, mask_camera, inter_occs[i])
                if self.inter_sup_non_mask:
                    mask_camera=None
                loss_occ =self.TwoPart_loss(inter_occs[i],voxel_semantics, mask_camera,weight,weight_gt)
                loss_occ = {key+str(i):value for key,value in loss_occ.items()}
                losses.update(loss_occ)
        return losses
