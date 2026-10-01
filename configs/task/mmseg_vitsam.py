# UPerNet ViT-SAM (mmpretrain.ViTSAM, 1024 px), self-contained: flattened (mmengine Config.fromfile, `_base_` resolved) from
#   third_party/Sensaug/sensaug/custom_configs/vitsam_cityscapes_1024.py
# which Sensaug uses for `--use-foundation-backbone` on cityscapes (train.py build_config).
#
# Not a complete experiment. tasks/mmseg/config.py fills in, from the composed dropaug
# config: the dataloaders, num_classes and crop size (dataset/), the GPU augmentation and its
# host (host/, vocab/), the probe hooks (probe/, selector/), max_iters and the intervals
# (task/mmseg_vitsam.yaml), the seed and the work dir. Edit the model or the optimizer here;
# edit everything else in the yaml groups.
#
# Sensaug only uses this file for cityscapes; on any other dataset the flag is ignored and the regular
# per-backbone pick applies.
#
# Deviations from the source (all deliberate):
#   - model.pretrained is None, as Sensaug's build_config forces (`cfg.model.pretrained = None`).
#   - data_preprocessor.size is dropped (the crop comes from the dataset config) and num_classes is
#     set to 19 (cityscapes); the builder overwrites both from the dataset.
#   - custom_imports (mmpretrain.models) is added: the source has none and relies on Sensaug's train.py
#     importing mmpretrain, but `mmpretrain.ViTSAM` needs it registered.
#   - the SAM weights load through backbone.init_cfg (kept).

custom_imports = dict(imports="mmpretrain.models", allow_failed_imports=False)

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
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=256,
        in_index=2,
        loss_decode=dict(loss_weight=0.4, type="CrossEntropyLoss", use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type="BN"),
        num_classes=19,
        num_convs=1,
        type="FCNHead",
    ),
    backbone=dict(
        type="mmpretrain.ViTSAM",
        arch="base",
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth",
            prefix="backbone.",
        ),
        out_indices=(3, 7, 11, 12),
        out_type="raw",
    ),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        loss_decode=dict(loss_weight=1.0, type="CrossEntropyLoss", use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type="BN"),
        num_classes=19,
        pool_scales=(1, 2, 3, 6),
        type="UPerHead",
    ),
    neck=dict(
        type="MultiLevelNeck",
        in_channels=[256, 256, 256],
        out_channels=256,
        scales=[2, 1, 0.5],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
    pretrained=None,
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        ),
    ),
)

# `end=max_iters` is rewritten to the yaml's max_iters.
max_iters = 80000
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-06, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
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
