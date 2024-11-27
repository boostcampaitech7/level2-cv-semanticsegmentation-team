_base_ = [
    'mmsegext::_base_/datasets/ade20k_512_tta_without_ratio.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py'
]
default_scope = 'mmseg'
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(MulticlassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # input: (N, C, H, W)
        # target: (N, H, W)
        N, C, H, W = input.shape
        
        # 원-핫 인코딩
        target_one_hot = F.one_hot(target, C).permute(0, 3, 1, 2).float()
        
        # 소프트맥스 적용
        input_softmax = F.softmax(input, dim=1)
        
        # 평탄화
        input_flat = input_softmax.view(N, C, -1)
        target_flat = target_one_hot.view(N, C, -1)
        
        # 교집합 계산
        intersection = torch.sum(input_flat * target_flat, dim=2)
        
        # Dice 계수 계산
        denominator = torch.sum(input_flat, dim=2) + torch.sum(target_flat, dim=2)
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        
        # 클래스별 Dice Loss 계산
        loss = 1 - dice
        
        # ignore_index 처리
        if self.ignore_index is not None:
            loss = loss[:, torch.arange(C) != self.ignore_index]
        
        # 가중치 적용 (옵션)
        if self.weight is not None:
            loss = loss * self.weight
        
        # 평균 계산
        return loss.mean()
'''
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(512, 512),
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='/data/ephemeral/home/wind/mmseg-extension/configs/vit_comer/deit_small_patch16_224-cd65a155.pth',
    backbone=dict(
        type='ext-ViTCoMer',
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        dim_ratio=1.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[
            False, False, False, False, False, False, False, False, False,
            False, False, False
        ],
        window_size=[
            None, None, None, None, None, None, None, None, None, None, None,
            None
        ]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=29,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=29,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    constructor='ext-LayerDecayOptimizerConstructorViTAdapter',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95),
)
# learning policy
param_scheduler = [
    # 线性学习率预热调度器
    dict(type='LinearLR',
         start_factor=1e-6,
         by_epoch=False,  # 按迭代更新学习率
         begin=0,
         end=1500),  # 预热前 50 次迭代
    # 主学习率调度器
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False)
]
