# UPerNet Swin-B (window7, IN-1k pretrain), self-contained: flattened (mmengine Config.fromfile, `_base_` resolved) from
#   third_party/Sensaug/sensaug/custom_configs/mmseg/swin/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py
# which Sensaug reaches by the FALLBACK path for (backbone=swin, dataset=cityscapes): there is no
# *cityscapes* config in `swin/`, so build_config takes the first natural-sorted non-r101 config,
# merges _base_/datasets/cityscapes.py and _base_/schedules/schedule_320k.py into it, and keeps the
# original optim_wrapper. The schedule below is therefore schedule_320k's (PolyLR 320k, eta_min 1e-4,
# power 0.9), the model and optimizer are the source's.
#
# Not a complete experiment. tasks/mmseg/config.py fills in, from the composed dropaug
# config: the dataloaders, num_classes and crop size (dataset/), the GPU augmentation and its
# host (host/, vocab/), the probe hooks (probe/, selector/), max_iters and the intervals
# (task/mmseg_swin_base.yaml), the seed and the work dir. Edit the model or the optimizer here;
# edit everything else in the yaml groups.
#
# On these datasets build_config picks a different file or path, and
# configs/recipe/<task>_<dataset>.yaml turns this task into it:
#   ade20k: swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py [direct]
#
# Deviations from the source (all deliberate):
#   - model.pretrained is None, as Sensaug's build_config forces (`cfg.model.pretrained = None`).
#   - data_preprocessor.size is dropped (the crop comes from the dataset config) and num_classes is
#     set to 19 (cityscapes); the builder overwrites both from the dataset.
#   - num_classes is 150 in the source (ade20k); 19 here.

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
        type="SwinTransformer",
        pretrain_img_size=224,
        embed_dims=128,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN", requires_grad=True),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth",
        ),
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=512,
        in_index=2,
        channels=256,
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
    optimizer=dict(type="AdamW", lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        ),
    ),
)

# The builder sets train_cfg.max_iters from the yaml but leaves param_scheduler alone (as
# Sensaug did), so a yaml max_iters other than this one doesn't move `end`.
max_iters = 320000
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
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=32000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

train_dataloader = dict(
    batch_size=4,
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
        type="CheckpointHook", by_epoch=False, interval=32000, save_best="mIoU", max_keep_ckpts=3
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
