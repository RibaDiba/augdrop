# UPerNet MAE ViT-B/16, self-contained: flattened (mmengine Config.fromfile, `_base_` resolved) from
#   third_party/Sensaug/sensaug/custom_configs/mmseg/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512-ms.py
# which Sensaug reaches by the FALLBACK path for (backbone=mae, dataset=cityscapes): there is no
# *cityscapes* config in `mae/`, so build_config takes the first natural-sorted non-r101 config,
# merges _base_/datasets/cityscapes.py and _base_/schedules/schedule_320k.py into it, and keeps the
# original optim_wrapper. The schedule below is therefore schedule_320k's (PolyLR 320k, eta_min 1e-4,
# power 0.9), the model and optimizer are the source's.
#
# Not a complete experiment. tasks/mmseg/config.py fills in, from the composed dropaug
# config: the dataloaders, num_classes and crop size (dataset/), the GPU augmentation and its
# host (host/, vocab/), the probe hooks (probe/, selector/), max_iters and the intervals
# (task/mmseg_mae_base.yaml), the seed and the work dir. Edit the model or the optimizer here;
# edit everything else in the yaml groups.
#
# Parity with Sensaug breaks on these datasets, where build_config picks a different file or path:
#   ade20k: mae-base_upernet_8xb2-amp-160k_ade20k-512x512-ms.py [direct]
#
# Deviations from the source (all deliberate):
#   - model.pretrained is None, as Sensaug's build_config forces (`cfg.model.pretrained = None`). The source named './pretrain/mae_pretrain_vit_base_mmcls.pth', which is never loaded; only `init_cfg` checkpoints below are.
#   - data_preprocessor.size is dropped (the crop comes from the dataset config) and num_classes is
#     set to 19 (cityscapes); the builder overwrites both from the dataset.
#   - num_classes is 150 in the source (ade20k); 19 here.
#   - The source's legacy `fp16 = dict(loss_scale="dynamic")` is dropped: mmengine ignores it, and the optim_wrapper is a plain OptimWrapper, so this trains in fp32 (as in Sensaug) despite `amp` in the file name.

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
        type="MAE",
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=[3, 5, 7, 11],
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type="LN", eps=1e-06),
        act_cfg=dict(type="GELU"),
        norm_eval=False,
        init_values=1.0,
    ),
    neck=dict(type="Feature2Pyramid", embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type="UPerHead",
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
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
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(341, 341)),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor="LayerDecayOptimizerConstructor",
)

# `end=max_iters` is rewritten to the yaml's max_iters.
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
