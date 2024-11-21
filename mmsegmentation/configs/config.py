_base_ = [
    '/data/ephemeral/home/mmsegmentation/configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py',
]

crop_size = (1024, 1024)
data_preprocessor = dict(
    _delete_=True,  # 기존 설정 삭제
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255)

# 모델 수정
model = dict(
    type='EncoderDecoderWithoutArgmax',
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='SegformerHeadWithoutAccuracy',
        num_classes=29,
        loss_decode=dict(
            _delete_=True,
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)

# 학습 설정
train_cfg = dict(
    _delete_=True,  # 기존 설정 삭제
    type='EpochBasedTrainLoop',
    max_epochs=1
)

default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=True),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, max_keep_ckpts=1, save_best='DiceMetric'))