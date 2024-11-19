_base_ = [
    '/data/ephemeral/home/mmsegmentation/configs/knet/knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512.py'
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
    decode_head=dict(
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',  # `type` 키를 추가
                num_classes=29,  # 새로운 클래스 수
                kernel_updator_cfg=dict(
                    type='KernelUpdator'
                )) for _ in range(3)  # 모든 스테이지에서 업데이트 보장
        ],
        kernel_generate_head=dict(
            type='FCNHead',  # `type` 키를 추가
            num_classes=29  # kernel_generate_head의 클래스 수 업데이트
        ),
    ),
    auxiliary_head=dict(
        type='FCNHead',  # `type` 키를 추가
        num_classes=29  # 보조 헤드의 클래스 수 업데이트
    ),
)


# 데이터 설정
train_dataloader = dict(
    _delete_=True,  # 기존 설정 삭제
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='ADE20KDataset',
        data_root='/data/ephemeral/home/data/ade20k_format',
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training',
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomRotate', prob=0.5, degree=(-30, 30)),
            dict(type='PackSegInputs'),
        ],
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    _delete_=True,  # 기존 설정 삭제
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type='ADE20KDataset',
        data_root='/data/ephemeral/home/data/ade20k_format',
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation',
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='PackSegInputs'),
        ],
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = val_dataloader

# 평가 지표 설정 (val_evaluator)
val_evaluator = dict(
    type='IoUMetric',  # Dice 지표 사용
    iou_merics=['mDice'],    # 클래스별 Dice를 평균
    ignore_index=None   # 무시할 클래스가 없는 경우
)
test_evaluator = val_evaluator

# 학습 설정
train_cfg = dict(
    _delete_=True,  # 기존 설정 삭제
    type='EpochBasedTrainLoop',
    max_epochs=1
)
# 평가 시 출력 크기를 맞추는 설정 추가
test_cfg=dict(
    _delete_=True,  # 기존 설정 삭제
    resize=True,  # 모델 출력 크기를 레이블 크기에 맞게 보강
    size=(1024, 1024),  # 평가 시 출력 크기
    mode="whole"  # 전체 이미지를 한 번에 처리
)

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
