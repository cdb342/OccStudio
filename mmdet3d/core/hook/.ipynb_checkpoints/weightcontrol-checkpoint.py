# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['WeightControlHook']


@HOOKS.register_module()
class WeightControlHook(Hook):
    """ """

    def __init__(self, warmup_iters=200, weight=0.9):
        super().__init__()
        self.warmup_iters=200
        self.weight=0.9
    # def set_temporal_flag(self, runner, flag):
    #     if is_parallel(runner.model.module):
    #         runner.model.module.module.with_prev=flag
    #     else:
    #         runner.model.module.with_prev = flag

    def set_temporal_flag_v2(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.semantic_prototype_decay=flag
        else:
            runner.model.module.semantic_prototype_decay = flag

    # def before_run(self, runner):
    #     self.set_temporal_flag(runner, False)
    #     if self.temporal_start_iter>0:
    #         self.set_temporal_flag_v2(runner, False)

    # def before_train_epoch(self, runner):
    #     if runner.epoch > self.temporal_start_epoch and self.temporal_start_iter<0:
    #         self.set_temporal_flag(runner, True)

    def after_train_iter(self, runner):
 
        curr_step = runner.iter
        # import pdb; pdb.set_trace()
        if curr_step <= self.warmup_iters:
            weight=curr_step/self.warmup_iters*self.weight

            self.set_temporal_flag_v2(runner, weight)
# 
