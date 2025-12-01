from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
import math
__all__ = ['FusionRateControlHook']


@HOOKS.register_module()
class FusionRateControlPoseHook(Hook):
    """ """

    def __init__(self, temporal_end_iter=1, temporal_start_iter=-1):
        super().__init__()
        self.temporal_start_iter = temporal_start_iter
        self.temporal_end_iter=temporal_end_iter

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.pose_weight=flag
        else:
            runner.model.module.pose_weight = flag

    def before_train_iter(self, runner):
        
        curr_step = runner.iter

        if curr_step>self.temporal_start_iter:
            if curr_step>self.temporal_end_iter:
                zz=1.
            else:
                zz=1-(math.cos((curr_step-self.temporal_start_iter)/(self.temporal_end_iter-self.temporal_start_iter)*math.pi)+1)/2
            self.set_temporal_flag(runner,zz)
        if curr_step==self.temporal_start_iter or curr_step==self.temporal_end_iter:
            print('pose_weight:',runner.model.module.pose_weight)
            


