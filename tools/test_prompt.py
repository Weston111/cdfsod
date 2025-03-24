# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import defaultdict
import itertools
import json
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
import ipdb

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    # parser.add_argument('--prompt', type=str, default='')
    parser.add_argument(
        '--class-names',
        nargs='+',
        help='List of class names, must match the order of descriptions')
    parser.add_argument('--descriptions', nargs='+', action='append', help='Descriptions for each class')
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=5,
        help='Maximum number of combinations to test per class. Set to -1 for all combinations')
    parser.add_argument(
        '--use-prefix',
        action='store_true',
        help='Use descriptions as prefix instead of direct replacement')
    parser.add_argument(
        '--use-suffix',
        action='store_true',
        help='Use descriptions as suffix instead of direct replacement')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def get_all_combinations(descriptions_list, max_combinations):
    """生成所有可能的描述组合，每个类别可以选择一个描述词或不选"""
    if not descriptions_list:
        return {}
    
    all_combinations = {}
    for i, descs in enumerate(descriptions_list):
        # 为每个类别生成所有可能的组合
        combinations = []
        
        # 首先添加不选择任何描述词的情况
        combinations.append(tuple())  # 空元组表示不选择任何描述词
        
        # 然后添加单个描述词的情况
        for desc in descs:
            combinations.append((desc,))
        
        # 如果max_combinations > 0，则限制组合数量
        if max_combinations > 0 and len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
        
        all_combinations[i] = combinations
    
    return all_combinations

def create_caption_prompt(class_names, combination_dict, use_prefix, use_suffix):
    """根据组合创建caption_prompt字典"""
    caption_prompt = {}
    
    for class_idx, combinations in combination_dict.items():
        if class_idx >= len(class_names):
            continue
            
        class_name = class_names[class_idx]
        desc_text = " ".join(combinations)
        
        caption_prompt[class_name] = {}
        
        if use_prefix:
            caption_prompt[class_name]['prefix'] = desc_text + " "
        elif use_suffix:
            caption_prompt[class_name]['suffix'] = " " + desc_text
        else:
            caption_prompt[class_name]['name'] = desc_text
            
    return caption_prompt

def run_test(config_file, caption_prompt, work_dir,args):
    """使用给定的caption_prompt运行测试并返回结果"""
    cfg = Config.fromfile(config_file)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # 修改配置以使用当前的caption_prompt
    cfg.caption_prompt = caption_prompt
    cfg.load_from = args.checkpoint
    # ipdb.set_trace()
    # 确保dataset_prompt和其他相关配置使用caption_prompt
    if hasattr(cfg, 'dataset_prompt'):
        cfg.dataset_prompt.caption_prompt = caption_prompt
    
    if hasattr(cfg, 'val_dataloader') and hasattr(cfg.val_dataloader, 'dataset'):
        cfg.val_dataloader.dataset.caption_prompt = caption_prompt
    
    if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset'):
        cfg.test_dataloader.dataset.caption_prompt = caption_prompt
    
    # 设置工作目录和GPU
    cfg.work_dir = work_dir
    
    # 创建临时检查点目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 创建Runner并运行测试
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    
    return metrics


def main():
    args = parse_args()
    # 处理输入的描述词列表
    descriptions_list = args.descriptions
    if not descriptions_list or not args.class_names:
        raise ValueError("必须提供描述词列表和类别名称")
    
    if len(descriptions_list) != len(args.class_names):
        raise ValueError("描述词列表数量必须与类别名称数量一致")
    # 打印输入信息用于调试
    print(f"类别名称: {args.class_names}")
    print(f"描述词列表: {descriptions_list}")
    print(f"max_combinations: {args.max_combinations}")
    print(f"use_prefix: {args.use_prefix}")
    print(f"use_suffix: {args.use_suffix}")
    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()
    base_work_dir = args.work_dir
    os.makedirs(base_work_dir, exist_ok=True)
    class_combinations = get_all_combinations(descriptions_list, args.max_combinations)
    # 打印生成的组合
    print("\n为每个类别生成的组合:")
    for class_idx, combinations in class_combinations.items():
        class_name = args.class_names[class_idx] if class_idx < len(args.class_names) else f"未知类别{class_idx}"
        print(f"{class_name}: {combinations}")

    # ipdb.set_trace()
    all_class_combinations = []
    for class_idx in range(len(args.class_names)):
        if class_idx in class_combinations:
            all_class_combinations.append(class_combinations[class_idx])
        else:
            all_class_combinations.append([tuple()])  # 没有描述词的默认空组合
        # 生成所有类别组合的笛卡尔积
    all_combinations = list(itertools.product(*all_class_combinations))
    print(f"\n共生成 {len(all_combinations)} 个测试组合:")
    # ipdb.set_trace()
    for i, combination in enumerate(all_combinations):
        combination_dict = {class_idx: comb for class_idx, comb in enumerate(combination)}
        caption_prompt = create_caption_prompt(
            args.class_names, combination_dict, args.use_prefix, args.use_suffix)
        print(f"组合 {i}: {caption_prompt}")

    results = []

    for i, combination in enumerate(all_combinations):
        # 创建当前组合的caption_prompt
        combination_dict = {class_idx: comb for class_idx, comb in enumerate(combination)}
        caption_prompt = create_caption_prompt(
            args.class_names, combination_dict, args.use_prefix, args.use_suffix)
        
        # 创建当前测试的工作目录
        current_work_dir = os.path.join(base_work_dir, f"test_{i}")
        os.makedirs(current_work_dir, exist_ok=True)
        # 运行测试
        try:
            print(f"开始测试组合",caption_prompt)
            metrics = run_test(args.config, caption_prompt, current_work_dir,args)
            
            # 记录结果
            result = {
                "combination_id": i,
                "caption_prompt": caption_prompt,
                "metrics": metrics
            }
            results.append(result)
            
            # 打印当前结果
            print(f"\n组合 {i}:")
            print(f"描述词: {caption_prompt}")
            print(f"mAP: {metrics.get('coco/bbox_mAP', 'N/A')}")
            
            # 保存中间结果
            with open(os.path.join(base_work_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"测试组合 {i} 时出错: {e}")
    
    # 如果没有结果，提前退出
    if not results:
        print("没有成功的测试结果，请检查日志了解详情。")
        return
    
    # 找出最佳的组合
    best_result = None
    best_score = -1
    
    for result in results:
        score = result["metrics"].get("coco/bbox_mAP", 0)
        if score > best_score:
            best_score = score
            best_result = result
    
    # 输出最佳结果
    if best_result:
        print("\n最佳描述组合:")
        print(f"组合 ID: {best_result['combination_id']}")
        print(f"描述词: {best_result['caption_prompt']}")
        print(f"mAP: {best_score}")
        
        # 生成每个类别的最佳结果
        class_best_scores = defaultdict(lambda: -1)
        class_best_results = {}
        
        for result in results:
            metrics = result["metrics"]
            for key, value in metrics.items():
                if key.startswith("coco/bbox_mAP_") and not key.endswith("50") and not key.endswith("75"):
                    # 提取类别名称
                    class_name = key.replace("coco/bbox_mAP_", "")
                    if value > class_best_scores[class_name]:
                        class_best_scores[class_name] = value
                        class_best_results[class_name] = result
        
        print("\n各类别最佳描述组合:")
        for class_name, result in class_best_results.items():
            print(f"{class_name}: {result['caption_prompt'].get(class_name, 'N/A')} (mAP: {class_best_scores[class_name]})")
    
    # 保存最终结果
    with open(os.path.join(base_work_dir, "final_results.json"), "w") as f:
        final_results = {
            "all_results": results,
            "best_result": best_result,
            "class_best_results": {k: v["caption_prompt"].get(k, None) for k, v in class_best_results.items()} if best_result else {}
        }
        json.dump(final_results, f, indent=2)
    
    print(f"\n所有结果已保存到 {os.path.join(base_work_dir, 'final_results.json')}")


    # load config
    # cfg = Config.fromfile(config_file)
    # cfg.launcher = args.launcher
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # # work_dir is determined in this priority: CLI > segment in file > filename
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get('work_dir', None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     cfg.work_dir = osp.join('./work_dirs',
    #                             osp.splitext(osp.basename(args.config))[0])

    # cfg.load_from = args.checkpoint

    # # build the runner from config
    # if 'runner_type' not in cfg:
    #     # build the default runner
    #     runner = Runner.from_cfg(cfg)
    # else:
    #     # build customized runner from the registry
    #     # if 'runner_type' is set in the cfg
    #     runner = RUNNERS.build(cfg)

    # # start testing
    # runner.test()


if __name__ == '__main__':
    main()
