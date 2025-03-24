# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--show-per-class',
        action='store_true',
        help='Show per-class evaluation results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    predictions = mmengine.load(args.pkl_results)
    # 确保评估器配置为显示每个类别的指标
    if args.show_per_class:
        if isinstance(cfg.val_evaluator, list):
            for evaluator_cfg in cfg.val_evaluator:
                if 'classwise' in evaluator_cfg:
                    evaluator_cfg['classwise'] = True
                else:
                    evaluator_cfg['classwise'] = True
        else:
            if 'classwise' in cfg.val_evaluator:
                cfg.val_evaluator['classwise'] = True
            else:
                cfg.val_evaluator['classwise'] = True


    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo
    eval_results = evaluator.offline_evaluate(predictions)
    print(eval_results)
    # 打印每个类别的评估结果（如果存在）
    # if args.show_per_class and 'classwise_results' in eval_results:
    #     print("\n每个类别的评估结果:")
    #     class_names = dataset.metainfo.get('classes', None)
    #     classwise_results = eval_results['classwise_results']
        
    #     for i, (class_id, metrics) in enumerate(classwise_results.items()):
    #         class_name = class_names[i] if class_names else f"类别 {class_id}"
    #         print(f"\n{class_name}:")
    #         for metric_name, value in metrics.items():
    #             print(f"  {metric_name}: {value:.4f}")


if __name__ == '__main__':
    main()
