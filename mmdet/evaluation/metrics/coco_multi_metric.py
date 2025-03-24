# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls
import ipdb

@METRICS.register_module()
class Coco_Multi_Metric(BaseMetric):
    """Extended COCO evaluation metric that can evaluate on multiple json files.

    This metric allows evaluating the same predictions on different ground truth 
    annotation files, such as evaluating performance on all classes, base classes,
    and novel classes separately in a single run.

    Args:
        ann_files (List[dict], optional): List of dictionaries, each containing:
            - path (str): Path to the coco format annotation file
            - prefix (str): Prefix to add to the metric names from this file
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_files: Optional[List[dict]] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.backend_args = backend_args

        # Initialize a dictionary to store multiple COCO API instances
        self.coco_apis = {}
        
        # If ann_files is not provided, we'll use a single API based on dataset annotations
        if ann_files is not None:
            for ann_file_info in ann_files:
                file_path = ann_file_info['path']
                api_name = ann_file_info.get('prefix', osp.basename(file_path).split('.')[0])
                
                with get_local_path(file_path, backend_args=self.backend_args) as local_path:
                    coco_api = COCO(local_path)
                    if sort_categories:
                        # Sort categories if needed
                        cats = coco_api.cats
                        sorted_cats = {i: cats[i] for i in sorted(cats)}
                        coco_api.cats = sorted_cats
                        categories = coco_api.dataset['categories']
                        sorted_categories = sorted(
                            categories, key=lambda i: i['id'])
                        coco_api.dataset['categories'] = sorted_categories
                    
                    self.coco_apis[api_name] = coco_api
        
        # Handle case where we'll convert GT from dataset
        if not self.coco_apis:
            self.coco_apis['all'] = None

        # Handle dataset lazy init - maintain separate cat_ids and img_ids for each COCO API
        self.cat_ids = {k: None for k in self.coco_apis.keys()}
        self.img_ids = {k: None for k in self.coco_apis.keys()}

    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         coco_api: COCO,
                         img_ids: List[int],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.

        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            coco_api (COCO): COCO API instance to use for evaluation.
            img_ids (List[int]): Image IDs to evaluate on.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(img_ids)):
            ann_ids = coco_api.get_ann_ids(img_ids=img_ids[i])
            ann_info = coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        # Since we need to evaluate on multiple annotation files with the same
        # predictions, we'll use the first COCO API's category IDs
        first_api_name = next(iter(self.coco_apis.keys()))
        if self.cat_ids[first_api_name] is None:
            # Initialize cat_ids if not already done
            coco_api = self.coco_apis[first_api_name]
            if coco_api is not None:
                self.cat_ids[first_api_name] = coco_api.get_cat_ids(
                    cat_names=self.dataset_meta['classes'])
            else:
                # If we don't have a COCO API yet (will be created from dataset),
                # use sequential IDs based on dataset classes
                self.cat_ids[first_api_name] = list(range(len(self.dataset_meta['classes'])))
        
        # Use the category IDs from the first API for conversion
        cat_ids = self.cat_ids[first_api_name]
        
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                     outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            
            # If we need to create GT from dataset
            for api_name in self.coco_apis:
                if self.coco_apis[api_name] is None:
                    # We only need to add GT annotations once (for the first None API)
                    assert 'instances' in data_sample, \
                        'ground truth is required for evaluation when ' \
                        '`ann_file` is not provided'
                    gt['anns'] = data_sample['instances']
                    break
                    
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        # Handle case where we need to create a COCO API from the dataset
        for api_name in list(self.coco_apis.keys()):
            if self.coco_apis[api_name] is None:
                # use converted gt json file to initialize coco api
                logger.info(f'Converting ground truth to coco format for {api_name}...')
                coco_json_path = self.gt_to_coco_json(
                    gt_dicts=gts, outfile_prefix=f"{outfile_prefix}_{api_name}")
                self.coco_apis[api_name] = COCO(coco_json_path)

        # Make sure we have category IDs and image IDs for each COCO API
        # for api_name, coco_api in self.coco_apis.items():
        #     if self.cat_ids[api_name] is None:
        #         self.cat_ids[api_name] = coco_api.get_cat_ids(
        #             cat_names=self.dataset_meta['classes'])
        #     if self.img_ids[api_name] is None:
        #         self.img_ids[api_name] = coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        # Evaluate on each annotation file
        for api_name, coco_api in self.coco_apis.items():
            logger.info(f'Evaluating on {api_name} annotations...')
            if self.cat_ids[api_name] is None:
                self.cat_ids[api_name] = coco_api.get_cat_ids(
                    cat_names=self.dataset_meta['classes'])
            if self.img_ids[api_name] is None:
                self.img_ids[api_name] = coco_api.get_img_ids()

            for metric in self.metrics:
                logger.info(f'Evaluating {metric} on {api_name}...')

                # Fast eval recall (if needed)
                if metric == 'proposal_fast':
                    ar = self.fast_eval_recall(
                        preds, self.proposal_nums, self.iou_thrs, 
                        coco_api, self.img_ids[api_name], logger=logger)
                    log_msg = []
                    for i, num in enumerate(self.proposal_nums):
                        eval_results[f'{api_name}_AR@{num}'] = ar[i]
                        log_msg.append(f'\n{api_name}_AR@{num}\t{ar[i]:.4f}')
                    log_msg = ''.join(log_msg)
                    logger.info(log_msg)
                    continue

                # Evaluate proposal, bbox and segm
                iou_type = 'bbox' if metric == 'proposal' else metric
                if metric not in result_files:
                    raise KeyError(f'{metric} is not in results')
                try:
                    predictions = load(result_files[metric])
                    predictions = [
                        pred for pred in predictions 
                        if pred['image_id'] in self.img_ids[api_name]
                    ]
                    if iou_type == 'segm':
                        # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                        # When evaluating mask AP, if the results contain bbox,
                        # cocoapi will use the box area instead of the mask area
                        # for calculating the instance area. Though the overall AP
                        # is not affected, this leads to different
                        # small/medium/large mask AP results.
                        for x in predictions:
                            x.pop('bbox')
                    coco_dt = coco_api.loadRes(predictions)

                except IndexError:
                    logger.error(
                        f'The testing results of {api_name} dataset is empty.')
                    break

                if self.use_mp_eval:
                    coco_eval = COCOevalMP(coco_api, coco_dt, iou_type)
                else:
                    coco_eval = COCOeval(coco_api, coco_dt, iou_type)

                coco_eval.params.catIds = self.cat_ids[api_name]
                coco_eval.params.imgIds = self.img_ids[api_name]
                coco_eval.params.maxDets = list(self.proposal_nums)
                coco_eval.params.iouThrs = self.iou_thrs

                # mapping of cocoEval.stats
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }
                metric_items = self.metric_items
                if metric_items is not None:
                    for metric_item in metric_items:
                        if metric_item not in coco_metric_names:
                            raise KeyError(
                                f'metric item "{metric_item}" is not supported')

                if metric == 'proposal':
                    coco_eval.params.useCats = 0
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    if metric_items is None:
                        metric_items = [
                            'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                            'AR_m@1000', 'AR_l@1000'
                        ]

                    for item in metric_items:
                        val = float(
                            f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                        eval_results[f'{api_name}_{item}'] = val
                else:
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    if self.classwise:  # Compute per-category AP
                        # Compute per-category AP
                        # from https://github.com/facebookresearch/detectron2/
                        precisions = coco_eval.eval['precision']
                        # precision: (iou, recall, cls, area range, max dets)
                        assert len(self.cat_ids[api_name]) == precisions.shape[2]

                        results_per_category = []
                        for idx, cat_id in enumerate(self.cat_ids[api_name]):
                            t = []
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            nm = coco_api.loadCats(cat_id)[0]
                            precision = precisions[:, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{nm["name"]}')
                            t.append(f'{round(ap, 3)}')
                            eval_results[f'{api_name}_{nm["name"]}_precision'] = round(ap, 3)

                            # indexes of IoU  @50 and @75
                            for iou in [0, 5]:
                                precision = precisions[iou, :, idx, 0, -1]
                                precision = precision[precision > -1]
                                if precision.size:
                                    ap = np.mean(precision)
                                else:
                                    ap = float('nan')
                                t.append(f'{round(ap, 3)}')

                            # indexes of area of small, median and large
                            for area in [1, 2, 3]:
                                precision = precisions[:, :, idx, area, -1]
                                precision = precision[precision > -1]
                                if precision.size:
                                    ap = np.mean(precision)
                                else:
                                    ap = float('nan')
                                t.append(f'{round(ap, 3)}')
                            results_per_category.append(tuple(t))

                        num_columns = len(results_per_category[0])
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = [
                            'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                            'mAP_m', 'mAP_l'
                        ]
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data = [headers]
                        table_data += [result for result in results_2d]
                        table = AsciiTable(table_data)
                        logger.info(f'\n{api_name} classwise results:\n' + table.table)

                    if metric_items is None:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                        ]

                    for metric_item in metric_items:
                        key = f'{api_name}_{metric}_{metric_item}'
                        val = coco_eval.stats[coco_metric_names[metric_item]]
                        eval_results[key] = float(f'{round(val, 3)}')

                    ap = coco_eval.stats[:6]
                    logger.info(f'{api_name}_{metric}_mAP_copypaste: {ap[0]:.3f} '
                                f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                                f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results