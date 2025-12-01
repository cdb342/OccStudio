from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
import math
__all__ = ['FusionRateControlHook']


@HOOKS.register_module()
class FusionRateControlDepthHook(Hook):
    """ """

    def __init__(self, temporal_end_iter=1, temporal_start_iter=-1):
        super().__init__()
        self.temporal_start_iter = temporal_start_iter
        self.temporal_end_iter=temporal_end_iter


    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            if hasattr(runner.model.module.module,'geometry_denoise_rate'):
                runner.model.module.module.geometry_denoise_rate=flag
            if hasattr(runner.model.module.module.depth_net,'geometry_denoise_rate'):
                runner.model.module.module.depth_net.geometry_denoise_rate=flag
            
            
        else:
            if hasattr(runner.model.module,'geometry_denoise_rate'):
                runner.model.module.geometry_denoise_rate = flag
            if hasattr(runner.model.module.depth_net,'geometry_denoise_rate'):
                runner.model.module.depth_net.geometry_denoise_rate=flag

    def before_train_iter(self, runner):
        curr_step = runner.iter
        if curr_step>self.temporal_start_iter:
            if curr_step>self.temporal_end_iter:
                zz=0.
            else:
                zz=(math.cos((curr_step-self.temporal_start_iter)/(self.temporal_end_iter-self.temporal_start_iter)*math.pi)+1)/2
            self.set_temporal_flag(runner,zz)


