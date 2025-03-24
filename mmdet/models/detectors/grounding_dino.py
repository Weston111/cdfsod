# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import ipdb
import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor
from torchvision.ops import box_iou

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 use_pseudo_labeling=False,
                 pseudo_label_cfg=dict(
                     score_thr=0.5,
                     iou_thr=0.5,
                     max_pseudo_instances_per_gt=3,
                     weight=0.5
                 ),
                 **kwargs) -> None:
        """Initialize the GroundingDINO detector.
        
        Args:
            language_model (dict): Config for language model.
            use_autocast (bool): Whether to use automatic mixed precision.
            use_pseudo_labeling (bool): Whether to use pseudo labeling during
                training. If True, the model will first predict pseudo labels
                with current parameters, filter them by score threshold and
                IoU threshold, and then add them to ground truth for training.
            pseudo_label_cfg (dict): Configuration for pseudo labeling.
                - score_thr (float): Threshold for filtering pseudo labels
                  based on confidence score. Default: 0.5.
                - iou_thr (float): IoU threshold to filter out pseudo labels
                  that overlap with existing ground truth boxes. Default: 0.5.
                - max_pseudo_instances_per_gt (int): Maximum number of pseudo
                  instances per ground truth instance. Default: 3.
                - weight (float): Weight for pseudo label loss. Default: 0.5.
        """
        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.use_pseudo_labeling = use_pseudo_labeling
        self.pseudo_label_cfg = pseudo_label_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        # if self.add_linear_layer:
        #     self.tunable_linear = torch.nn.Linear(self.language_model.language_backbone.body.language_dim, 1000, bias=False)
        #     self.tunable_linear.weight.data.fill_(0.0)
        # if self.prompt_learning:
        #     # 初始化可学习的提示向量
        #     self.prompt_embeddings = nn.Parameter(
        #         torch.zeros(self.num_prompts, 
        #                     self.language_model.language_backbone.body.language_dim))
        #     # 初始化为正态分布
        #     nn.init.normal_(self.prompt_embeddings, std=0.02)

    def apply_learnable_prompts(self, text_dict):
        """应用可学习的提示向量到文本特征中"""
        if not self.prompt_learning:
            return text_dict
            
        # 获取原始文本嵌入
        embedded = text_dict['embedded']  # [B, L, D]
        batch_size, seq_len, dim = embedded.shape
        
        # 对于每个批次中的样本，在文本嵌入的开头插入可学习的提示向量
        # 这里假设每个类别都使用相同的提示向量集合
        prompt_emb = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, P, D]
        
        # 更新文本掩码以适应新增的提示向量
        text_token_mask = text_dict['text_token_mask']  # [B, L]
        prompt_mask = torch.zeros(batch_size, self.num_prompts, 
                                 device=text_token_mask.device, 
                                 dtype=text_token_mask.dtype)  # [B, P]
        
        # 合并提示向量和原始文本嵌入
        new_embedded = torch.cat([prompt_emb, embedded], dim=1)  # [B, P+L, D]
        new_text_token_mask = torch.cat([prompt_mask, text_token_mask], dim=1)  # [B, P+L]
        
        # 更新位置编码
        position_ids = text_dict['position_ids']
        if position_ids is not None:
            # 为提示向量创建新的位置ID
            prompt_position_ids = torch.arange(
                self.num_prompts, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
            new_position_ids = torch.cat([prompt_position_ids, position_ids + self.num_prompts], dim=1)
            text_dict['position_ids'] = new_position_ids
        
        # 更新注意力掩码
        if 'masks' in text_dict:
            masks = text_dict['masks']
            if masks is not None:
                # 扩展注意力掩码以包含提示向量
                prompt_attention_mask = torch.zeros(
                    batch_size, self.num_prompts, seq_len + self.num_prompts,
                    device=masks.device, dtype=masks.dtype)
                new_masks = torch.cat([prompt_attention_mask, 
                                      torch.cat([torch.zeros(batch_size, seq_len, self.num_prompts,
                                                           device=masks.device, dtype=masks.dtype), 
                                               masks], dim=2)], dim=1)
                text_dict['masks'] = new_masks
        
        # 更新文本字典
        text_dict['embedded'] = new_embedded
        text_dict['text_token_mask'] = new_text_token_mask
        return text_dict
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def generate_pseudo_labels(self, batch_inputs: Tensor,
                               batch_data_samples: SampleList):
        """Generate pseudo labels by running prediction before training.
        
        此方法实现了模型训练前的伪标签生成流程：
        1. 首先将模型切换到评估模式，使用当前模型权重生成预测结果
        2. 对预测结果进行过滤：
           - 通过置信度阈值筛选高质量预测
           - 通过IoU阈值避免与真实标签重叠的预测
           - 限制每个真实框对应的最大伪标签数量
        3. 将过滤后的伪标签添加到真实标签中用于训练
        4. 为伪标签指定较低的损失权重，使模型更依赖真实标签而非伪标签
        
        通过这种方式，模型可以使用自己的预测扩充训练数据，提高对稀有类别的检测性能，
        同时保持对高置信度预测的关注。这种自我训练策略在半监督学习中很常见。
        
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples.
                
        Returns:
            list[:obj:`DetDataSample`]: The data samples with pseudo labels added.
        """
        # 切换到评估模式进行推理
        self.eval()
        with torch.no_grad():
            # 使用现有的predict方法获取预测结果
            pseudo_data_samples = self.predict(
                batch_inputs=batch_inputs,
                batch_data_samples=copy.deepcopy(batch_data_samples),
                rescale=True)
        # 切换回训练模式
        self.train()
        
        score_thr = self.pseudo_label_cfg.get('score_thr', 0.5)
        iou_thr = self.pseudo_label_cfg.get('iou_thr', 0.5)
        max_pseudo_per_gt = self.pseudo_label_cfg.get('max_pseudo_instances_per_gt', 3)
        
        for i, (data_sample, pseudo_sample) in enumerate(zip(batch_data_samples, pseudo_data_samples)):
            gt_instances = data_sample.gt_instances
            pseudo_instances = pseudo_sample.pred_instances
            
            # 如果没有预测到任何实例，跳过此样本
            if len(pseudo_instances) == 0:
                continue
                
            # 过滤掉低置信度的伪标签
            valid_inds = pseudo_instances.scores >= score_thr
            if not valid_inds.any():
                continue
            
            filtered_pseudo_instances = pseudo_instances[valid_inds]
            
            # 如果有真实标签，计算与真实标签的IoU以过滤掉与真实标签重叠的预测
            if len(gt_instances) > 0:
                pseudo_bboxes = filtered_pseudo_instances.bboxes
                gt_bboxes = gt_instances.bboxes
                iou_matrix = box_iou(pseudo_bboxes, gt_bboxes)
                
                # 对于每个真实框，找出与之IoU大于阈值的伪标签
                max_ious, _ = iou_matrix.max(dim=1)
                # 保留IoU小于阈值的预测（避免与真实框重叠）
                keep_inds = max_ious < iou_thr
                
                if not keep_inds.any():
                    continue
                
                filtered_pseudo_instances = filtered_pseudo_instances[keep_inds]
                
                # 如果过滤后没有伪标签，跳过
                if len(filtered_pseudo_instances) == 0:
                    continue
                    
                # 限制每个真实框最多使用的伪标签数量
                if max_pseudo_per_gt > 0 and len(gt_instances) > 0:
                    if len(filtered_pseudo_instances) > max_pseudo_per_gt * len(gt_instances):
                        # 根据置信度排序，取前N个
                        _, indices = filtered_pseudo_instances.scores.sort(descending=True)
                        top_k = max_pseudo_per_gt * len(gt_instances)
                        keep_indices = indices[:top_k]
                        filtered_pseudo_instances = filtered_pseudo_instances[keep_indices]
            
            # 将过滤后的伪标签添加到真实标签中
            if len(filtered_pseudo_instances) > 0:
                # 采用一个更简单的方法：
                # 1. 获取所有必要字段
                pseudo_bboxes = filtered_pseudo_instances.bboxes
                pseudo_labels = filtered_pseudo_instances.labels
                
                # 2. 将原始gt_instances克隆到新的实例中
                from mmengine.structures import InstanceData
                new_instances = InstanceData()
                
                # 3. 合并基本字段
                gt_num = len(gt_instances)
                pseudo_num = len(filtered_pseudo_instances)
                total_num = gt_num + pseudo_num
                
                new_instances.bboxes = torch.cat([gt_instances.bboxes, pseudo_bboxes], dim=0)
                new_instances.labels = torch.cat([gt_instances.labels, pseudo_labels], dim=0)
                
                # 4. 设置伪标签标志
                is_pseudo = torch.zeros(total_num, dtype=torch.bool, device=pseudo_bboxes.device)
                is_pseudo[gt_num:] = True
                new_instances.is_pseudo = is_pseudo
                
                # 5. 添加伪标签权重
                pseudo_weight = self.pseudo_label_cfg.get('weight', 0.5)
                weights = torch.ones(total_num, dtype=torch.float, device=pseudo_bboxes.device)
                weights[gt_num:] = pseudo_weight
                new_instances.loss_weights = weights
                
                # 6. 复制原始gt_instances中的其他字段
                for key in gt_instances.keys():
                    if key not in ['bboxes', 'labels', 'is_pseudo', 'loss_weights']:
                        attr_val = getattr(gt_instances, key)
                        if isinstance(attr_val, torch.Tensor) and attr_val.size(0) == gt_num:
                            # 如果是张量且第一维与gt_instances的数量匹配，创建对应的扩展张量
                            device = attr_val.device
                            dtype = attr_val.dtype
                            if attr_val.dim() > 1:
                                # 多维张量
                                shape = list(attr_val.shape)
                                shape[0] = pseudo_num
                                pseudo_val = torch.zeros(shape, dtype=dtype, device=device)
                                # 合并
                                setattr(new_instances, key, torch.cat([attr_val, pseudo_val], dim=0))
                            else:
                                # 一维张量
                                pseudo_val = torch.zeros(pseudo_num, dtype=dtype, device=device)
                                setattr(new_instances, key, torch.cat([attr_val, pseudo_val], dim=0))
                        else:
                            # 如果不是张量或形状不匹配，直接复制
                            setattr(new_instances, key, attr_val)
                
                # 7. 设置新的gt_instances
                data_sample.gt_instances = new_instances
                
        return batch_data_samples

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # 如果启用了伪标签训练，先生成伪标签
        if self.use_pseudo_labeling and self.training:
            batch_data_samples = self.generate_pseudo_labels(batch_inputs, batch_data_samples)
            
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for i, (token_positive, text_prompt, gt_label) in enumerate(zip(
                    tokens_positive, text_prompts, gt_labels)):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                
                # 检查是否有伪标签
                if self.use_pseudo_labeling and self.training and hasattr(batch_data_samples[i].gt_instances, 'is_pseudo'):
                    is_pseudo = batch_data_samples[i].gt_instances.is_pseudo
                    new_tokens_positive = []
                    
                    # 处理真实标签
                    for j, label in enumerate(gt_label):
                        if not is_pseudo[j]:  # 真实标签
                            if label.item() < len(token_positive):
                                new_tokens_positive.append(token_positive[label.item()])
                            else:
                                # 如果标签超出范围，使用第一个token_positive
                                # 这种情况一般不应该发生，但为了健壮性添加
                                new_tokens_positive.append(token_positive[0])
                        else:  # 伪标签
                            # 对于伪标签，我们使用其检测到的类别对应的token_positive
                            if label.item() < len(token_positive):
                                new_tokens_positive.append(token_positive[label.item()])
                            else:
                                # 如果标签超出范围，使用第一个token_positive
                                new_tokens_positive.append(token_positive[0])
                else:
                    # 原始处理方式
                    new_tokens_positive = [
                        token_positive[label.item()] for label in gt_label
                    ]
                    
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                
                for i, gt_label in enumerate(gt_labels):
                    # 检查是否有伪标签
                    if self.use_pseudo_labeling and self.training and hasattr(batch_data_samples[i].gt_instances, 'is_pseudo'):
                        is_pseudo = batch_data_samples[i].gt_instances.is_pseudo
                        new_tokens_positive = []
                        
                        # 为每个标签(包括伪标签)找到对应的tokens_positive
                        for j, label in enumerate(gt_label):
                            if label.item() < len(tokens_positive):
                                new_tokens_positive.append(tokens_positive[label.item()])
                            else:
                                # 标签超出范围时使用第一个
                                new_tokens_positive.append(tokens_positive[0])
                    else:
                        # 原始处理方式
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for i, (text_prompt, gt_label) in enumerate(zip(text_prompts, gt_labels)):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                            
                    # 检查是否有伪标签
                    if self.use_pseudo_labeling and self.training and hasattr(batch_data_samples[i].gt_instances, 'is_pseudo'):
                        is_pseudo = batch_data_samples[i].gt_instances.is_pseudo
                        new_tokens_positive = []
                        
                        # 为每个标签(包括伪标签)找到对应的tokens_positive
                        for j, label in enumerate(gt_label):
                            if label.item() < len(tokens_positive):
                                new_tokens_positive.append(tokens_positive[label.item()])
                            else:
                                # 标签超出范围时使用第一个
                                new_tokens_positive.append(tokens_positive[0])
                    else:
                        # 原始处理方式
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        # text_dict = self.apply_learnable_prompts(text_dict)

        # if self.add_linear_layer:
        #     # print(text_dict['embedded'].shape)
        #     text_dict['embedded'] = text_dict['embedded'] + self.tunable_linear.weight[:text_dict['embedded'].size(1), :].unsqueeze(0)
        #     text_dict['hidden'] = text_dict['hidden'] + self.tunable_linear.weight[:text_dict['embedded'].size(1), :].unsqueeze(0)

        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
                    
            # 如果使用伪标签，为伪标签添加权重信息
            if self.use_pseudo_labeling and self.training and hasattr(data_samples.gt_instances, 'is_pseudo'):
                pseudo_weight = self.pseudo_label_cfg.get('weight', 0.5)
                is_pseudo = data_samples.gt_instances.is_pseudo
                
                # 创建权重张量：真实标签权重为1.0，伪标签权重为pseudo_weight
                weights = torch.ones_like(is_pseudo, dtype=torch.float)
                weights[is_pseudo] = pseudo_weight
                
                # 保存到实例中，bbox_head的loss计算中将使用这些权重
                data_samples.gt_instances.loss_weights = weights
                
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        # ipdb.set_trace()
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            # print(text_prompts)
            text_dict = self.language_model(list(text_prompts))
            # text_dict = self.apply_learnable_prompts(text_dict)
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
