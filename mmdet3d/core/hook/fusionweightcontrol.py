# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
import math
__all__ = ['FusionRateControlHook']


@HOOKS.register_module()
class FusionRateControlHook(Hook):
    """ """

    def __init__(self, temporal_end_iter=1, temporal_start_iter=-1):
        super().__init__()
        self.temporal_start_iter = temporal_start_iter
        self.temporal_end_iter=temporal_end_iter


    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            if hasattr(runner.model.module.module,'alocc_head'):
                if runner.model.module.module.alocc_head is not None:
                    runner.model.module.module.alocc_head.flow_gt_denoise_rate=flag
            # import pdb;pdb.set_trace()
        else:
            if hasattr(runner.model.module,'alocc_head'):
                if runner.model.module.alocc_head is not None:
                    runner.model.module.alocc_head.flow_gt_denoise_rate = flag
            


    def before_train_iter(self, runner):
        # import pdb;pdb.set_trace()
        curr_step = runner.iter
        if curr_step>self.temporal_start_iter:
            if curr_step>self.temporal_end_iter:
                zz=0.
            else:
                zz=(math.cos((curr_step-self.temporal_start_iter)/(self.temporal_end_iter-self.temporal_start_iter)*math.pi)+1)/2
            self.set_temporal_flag(runner,zz)


