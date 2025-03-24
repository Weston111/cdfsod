_base_ = '../../grounding_dino_swin-l_pretrain_all.py'
custom_imports = dict(
    imports=['mmdet_awp'],
    allow_failed_imports=False)
data_root = 'data/cdfsod/dataset3/'
class_name = ('dent','scratch','crack','glass shatter','lamp broken','tire flat',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))
randomness = dict(
    seed = 2092800512,
    diff_rank_seed=True
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    _delete_=True,
    dataset=dict(
        # _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations/10_shot.json',
        data_prefix=dict(img='train/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

max_epoch = 12

train_cfg = dict(_delete_=True,
    type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=max_epoch+1)

default_hooks = dict(
    # _delete_=True,
    checkpoint=dict(by_epoch=True,interval=max_epoch, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=20))
# train_cfg = dict(max_epochs=max_epoch, val_interval=1)



param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[20],
        gamma=0.1)
]

# 定义AWP配置
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='AWPOptimWrapperConstructor',  # 使用自定义的构造器
    optimizer=dict(type='AdamW', lr=0.0001),
    adv_param="weight",
    adv_lr=1,
    adv_eps=0.001,
    start_epoch=10,  # 从第5个epoch开始使用AWP
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        })
)

# 添加AWP钩子
custom_hooks = [
    dict(type='AWPHook')
]
log_processor = dict(by_epoch=True)