import yaml
import wandb
from mmengine.config import Config
from mmengine.runner import Runner

def train_with_wandb_sweep():
    # Sweep에서 하이퍼파라미터 가져오기
    sweep_config = wandb.config

    # Config 파일 로드
    cfg = Config.fromfile('configs/config.py')

    # Optimizer 설정 업데이트
    cfg.optim_wrapper.optimizer.lr = sweep_config.optimizer.lr
    cfg.optim_wrapper.optimizer.weight_decay = sweep_config.optimizer.weight_decay

    # Param Scheduler 설정 업데이트
    cfg.param_scheduler[0].end = sweep_config.param_scheduler_0_end
    cfg.param_scheduler[1].power = sweep_config.param_scheduler_1_power

    # Batch size 설정 업데이트
    cfg.train_dataloader.batch_size = sweep_config.batch_size

    # Best Checkpoint Hook 추가
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1,
        save_best='val/DiceMetric',  # 가장 좋은 Dice Metric 모델 저장
        rule='greater'               # 높은 값이 더 좋은 경우
    )

    # Runner 생성 및 학습 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

# YAML 파일 읽기
with open('configs/sweep_config.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

# Sweep ID 생성
sweep_id = wandb.sweep(sweep_config, project="semantic_segmentation/mmseg/segformer")

# Sweep 실행
wandb.agent(sweep_id, function=train_with_wandb_sweep)
