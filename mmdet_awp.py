import torch
from copy import deepcopy
from typing import Dict, List, Optional, Union
import torch.nn as nn
import inspect

from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapper
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS, OPTIMIZERS
from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


class AWP:
    """
    Implements weighted adversarial perturbation for MMDetection framework
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.0001):
        self.model = model
        # 处理可能的模型封装情况
        if is_model_wrapper(self.model):
            self.model = self.model.module
            
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self, loss_fn, data_batch):
        if self.adv_lr == 0:
            return None
        self._save()
        self._attack_step()
        
        # 模型在被扰动的权重情况下进行前向计算
        adv_loss = self.model.loss(data_batch)
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class AWPOptimWrapper(OptimWrapper):
    """优化器包装器，支持AWP对抗训练。"""
    
    def __init__(self,
                 optimizer,
                 adv_param="weight",
                 adv_lr=1.0,
                 adv_eps=0.0001,
                 start_epoch=1,
                 accumulative_counts=1,
                 clip_grad=None,
                 **kwargs):
        # 调用父类的构造函数
        # import ipdb; ipdb.set_trace()
        super().__init__(optimizer, accumulative_counts,clip_grad, **kwargs)
        self.awp = None
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.current_epoch = 0
        self.clip_grad = clip_grad
    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None,
            model=None) -> None:
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        if self.current_epoch >= self.start_epoch:
            if self.awp is None and model is not None:
                self.awp = AWP(
                    model, self.optimizer,
                    adv_param=self.adv_param,
                    adv_lr=self.adv_lr,
                    adv_eps=self.adv_eps
                )
                print("start awp")
                print("start awp")
                print("start awp")
            
            # 如果AWP已初始化，则应用AWP对抗训练
            if self.awp is not None and hasattr(model, 'loss'):
                data_batch = model.data_batch
                adv_loss = self.awp.attack_step(None, data_batch)
                if adv_loss is not None:
                    print("adv_loss",adv_loss)
                    self.backward(adv_loss)
                    self.awp.restore()
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)
    # def update_params(self, loss, model=None):
    #     """更新参数。"""
    #     self.backward(loss)
        
    #     # 如果到达开始AWP的epoch，且AWP尚未初始化
    #     if self.current_epoch >= self.start_epoch:
    #         if self.awp is None and model is not None:
    #             self.awp = AWP(
    #                 model, self.optimizer,
    #                 adv_param=self.adv_param,
    #                 adv_lr=self.adv_lr,
    #                 adv_eps=self.adv_eps
    #             )
            
    #         # 如果AWP已初始化，则应用AWP对抗训练
    #         if self.awp is not None and hasattr(model, 'loss'):
    #             data_batch = model.data_batch
    #             adv_loss = self.awp.attack_step(None, data_batch)
    #             if adv_loss is not None:
    #                 self.backward(adv_loss)
    #                 self.awp.restore()
        
    #     if self.clip_grad is not None:
    #         self.clip_grads(self.parameters)
            
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class AWPOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """AWP优化器包装器构造器。"""
    
    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        """初始化AWP优化器包装器构造器。
        
        Args:
            optim_wrapper_cfg (dict): 优化器包装器的配置。
            paramwise_cfg (dict, optional): 参数级别配置的字典。默认为None。
        """
        super().__init__(optim_wrapper_cfg, paramwise_cfg)
    
    def __call__(self, model: nn.Module) -> OptimWrapper:
        """根据配置构造OptimWrapper的实例。
        
        Args:
            model (nn.Module): 将被优化的模型。
            
        Returns:
            OptimWrapper: 优化器包装器实例。
        """
        if hasattr(model, 'module'):
            model = model.module

        # 获取原始optim_wrapper_cfg
        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        
        # 提取AWP特定参数
        adv_param = optim_wrapper_cfg.pop('adv_param', 'weight')
        adv_lr = optim_wrapper_cfg.pop('adv_lr', 1.0)
        adv_eps = optim_wrapper_cfg.pop('adv_eps', 0.0001)
        start_epoch = optim_wrapper_cfg.pop('start_epoch', 1)
        accumulative_counts = optim_wrapper_cfg.get('accumulative_counts', 1)
        clip_grad = optim_wrapper_cfg.get('clip_grad', None)
        
        # 复制优化器配置
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cls = self.optimizer_cfg['type']
        
        # 获取优化器的第一个参数名（通常是'params'）
        if isinstance(optimizer_cls, str):
            with OPTIMIZERS.switch_scope_and_registry(None) as registry:
                optimizer_cls = registry.get(self.optimizer_cfg['type'])
        first_arg_name = next(
            iter(inspect.signature(optimizer_cls).parameters))
        
        # 如果没有paramwise设置，直接使用全局设置
        if not self.paramwise_cfg:
            optimizer_cfg[first_arg_name] = model.parameters()
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # 设置参数级的学习率和权重衰减
            params = []
            self.add_params(params, model)
            optimizer_cfg[first_arg_name] = params
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        
        # 使用AWPOptimWrapper包装优化器
        optim_wrapper = AWPOptimWrapper(
            optimizer=optimizer,
            adv_param=adv_param,
            adv_lr=adv_lr,
            adv_eps=adv_eps,
            start_epoch=start_epoch,
            accumulative_counts=accumulative_counts,
            clip_grad=clip_grad,
        )
        
        return optim_wrapper


@HOOKS.register_module()
class AWPHook(Hook):
    """用于AWP的钩子，主要用于更新当前的epoch。"""
    
    def before_train_epoch(self, runner):
        """在每个训练epoch开始前更新当前epoch。"""
        for optimizer_wrapper in runner.optim_wrapper.values() if isinstance(
            runner.optim_wrapper, dict) else [runner.optim_wrapper]:
            if hasattr(optimizer_wrapper, 'current_epoch'):
                optimizer_wrapper.current_epoch = runner.epoch

