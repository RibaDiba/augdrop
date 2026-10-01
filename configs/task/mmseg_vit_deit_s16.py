# UPerNet ViT DeiT-S/16, self-contained: flattened (mmengine Config.fromfile, `_base_` resolved) from
#   third_party/Sensaug/sensaug/custom_configs/mmseg/vit/vit_deit-s16_upernet_8xb2-40k_cityscapes-512x512.py
# which is the config Sensaug's build_config picks for (backbone=vit, dataset=cityscapes):
# glob `vit/*cityscapes*.py`, natural-sorted, first.
#
# Not a complete experiment. tasks/mmseg/config.py fills in, from the composed dropaug
# config: the dataloaders, num_classes and crop size (dataset/), the GPU augmentation and its
# host (host/, vocab/), the probe hooks (probe/, selector/), max_iters and the intervals
# (task/mmseg_vit_deit_s16.yaml), the seed and the work dir. Edit the model or the optimizer here;
# edit everything else in the yaml groups.
#
# Parity with Sensaug breaks on these datasets, where build_config picks a different file or path:
#   ade20k: vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py [direct]
#   pascal_voc12: vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py [fallback path: schedule_320k + that dataset's base merged in]
#   loveda: vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py [fallback path: schedule_320k + that dataset's base merged in]
#
# Deviations from the source (all deliberate):
#   - model.pretrained is None, as Sensaug's build_config forces (`cfg.model.pretrained = None`). The source named 'pretrain/deit_small_patch16_224-cd65a155.pth', which is never loaded; only `init_cfg` checkpoints below are.
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
        type="VisionTransformer",
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        norm_cfg=dict(type="LN", eps=1e-06),
        act_cfg=dict(type="GELU"),
        norm_eval=False,
        interpolate_mode="bicubic",
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[384, 384, 384, 384],
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
        in_channels=384,
        in_index=3,
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
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        ),
    ),
)

# NOTE: the source's PolyLR ends at 160000 while training stops at 80000 iterations (kept as is).
max_iters = 80000
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-06, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
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
