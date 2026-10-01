# DeepLabV3+ R18-D8, self-contained: flattened (mmengine Config.fromfile, `_base_` resolved) from
#   third_party/Sensaug/sensaug/custom_configs/mmseg/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py
# which is the config Sensaug's build_config picks for (backbone=deeplabv3plus, dataset=cityscapes):
# glob `deeplabv3plus/*cityscapes*.py`, natural-sorted, first.
#
# Not a complete experiment. tasks/mmseg/config.py fills in, from the composed dropaug
# config: the dataloaders, num_classes and crop size (dataset/), the GPU augmentation and its
# host (host/, vocab/), the probe hooks (probe/, selector/), max_iters and the intervals
# (task/mmseg_deeplabv3plus_r18.yaml), the seed and the work dir. Edit the model or the optimizer here;
# edit everything else in the yaml groups.
#
# Parity with Sensaug breaks on these datasets, where build_config picks a different file or path:
#   ade20k: deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py [direct]
#   pascal_voc12: deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py [fallback path: schedule_320k + that dataset's base merged in]
#   loveda: deeplabv3plus_r18-d8_4xb4-80k_loveda-512x512.py [direct]
#
# Deviations from the source (all deliberate):
#   - model.pretrained is None, as Sensaug's build_config forces (`cfg.model.pretrained = None`). The source named 'open-mmlab://resnet18_v1c', which is never loaded; only `init_cfg` checkpoints below are.
#   - data_preprocessor.size is dropped (the crop comes from the dataset config) and num_classes is
#     set to 19 (cityscapes); the builder overwrites both from the dataset.

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
    pretrained=None,
    backbone=dict(
        type="ResNetV1c",
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 12, 24, 36),
        c1_in_channels=64,
        c1_channels=12,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
)

# `end=max_iters` is rewritten to the yaml's max_iters.
max_iters = 80000
param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False,
    ),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=8000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=None,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=None,
)
test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

default_scope = "mmseg"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=8000, save_best="mIoU", max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
