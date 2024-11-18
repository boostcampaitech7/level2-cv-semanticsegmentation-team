_base_ = [
    '/data/ephemeral/home/mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py',
    '/data/ephemeral/home/mmsegmentation/configs/_base_/default_runtime.py',
    '/data/ephemeral/home/mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]

# 모델 수정
model = dict(
    decode_head=dict(
        num_classes=29  # 클래스 수
    ),
    auxiliary_head=dict(
        num_classes=29  # 클래스 수
    )
)

# 데이터셋 설정
dataset_type = 'ADE20KDataset'
data_root = '/data/ephemeral/home/data/ade20k_format'

train_dataloader = dict(
    batch_size=2,  # Smoke Test를 위한 작은 배치 크기
    num_workers=1,  # 데이터 로드에 사용할 워커 수
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',  # ADE20K 포맷에 맞춘 이미지 경로
            seg_map_path='annotations/training',  # ADE20K 포맷에 맞춘 주석 경로
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomRotate', prob=0.5, degree=(-30, 30)),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',  # ADE20K 포맷에 맞춘 이미지 경로
            seg_map_path='annotations/validation',  # ADE20K 포맷에 맞춘 주석 경로
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',  # 테스트도 validation과 동일하게 설정
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

# 평가 지표 설정 (val_evaluator)
val_evaluator = dict(
    type='IoUMetric',  # Dice 지표 사용
    iou_merics=['mDice'],    # 클래스별 Dice를 평균
    ignore_index=None   # 무시할 클래스가 없는 경우
)

test_evaluator = val_evaluator

# 학습 설정
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='WandbLoggerHook',
        #      init_kwargs=dict(project='semantic_segmentation/mmseg/smoke-test'),
        #      interval=1,
        #      log_checkpoint=True,
        #      log_checkpoint_metadata=True,
        #      num_eval_images=10)
    ]
)

# 체크포인트 저장 설정
checkpoint_config = dict(
    by_epoch=True,
    max_keep_ckpts=1
)

# 평가 설정
evaluation = dict(
    interval=1,
    metric='Dice',
    save_best='Dice'
)
