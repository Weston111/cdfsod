from typing import Optional

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """在连续多个 epoch 性能下降时提前停止训练。
    
    Args:
        patience (int): 允许性能连续下降的最大 epoch 数
        metric (str): 用于监控的性能指标，例如 'coco/bbox_mAP'
        min_delta (float): 性能下降的最小阈值，低于此值视为发生下降
    """
    
    def __init__(self, patience=3, metric='coco/bbox_mAP', min_delta=0.0):
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.last_score = None  # 添加上一轮的分数记录
        
    def after_val_epoch(self, runner, metrics):
        """在每个验证 epoch 后检查性能。"""
        if metrics is None or self.metric not in metrics:
            return
            
        current_score = metrics[self.metric]
        
        # 首次运行时初始化
        if self.last_score is None:
            self.last_score = current_score
            self.best_score = current_score  # 仍然记录最佳分数用于日志
            return
            
        # 检查是否相比上一轮性能下降
        if current_score < self.last_score - self.min_delta:
            self.counter += 1
            runner.logger.info(
                f'性能下降 ({self.counter}/{self.patience}): '
                f'当前 {self.metric}={current_score:.4f}, '
                f'上一轮={self.last_score:.4f}, '
                f'最佳={self.best_score:.4f}')
        else:
            self.counter = 0  # 性能没有下降，重置计数器
            
        # 更新最佳分数（用于记录）
        if current_score > self.best_score:
            self.best_score = current_score
            
        # 更新上一轮分数
        self.last_score = current_score
            
        # 如果连续多个 epoch 性能下降，则停止训练
        if self.counter >= self.patience:
            runner.logger.info(
                f'触发早停: {self.metric} 连续 {self.patience} 个 epoch 性能下降')
            raise RuntimeError(
                f'早停: {self.metric} 连续 {self.patience} 个 epoch 性能下降')
