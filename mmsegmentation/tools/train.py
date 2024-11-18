import wandb
from mmengine.config import Config
from mmseg.apis import init_segmentor, train_segmentor

wandb.login(key="97ae30a3f47da7ce905379af5f6bfc8d2d4dd531")

# WandB Sweep 초기화
wandb.init(project="semantic_segmentation/mmseg/smoke-test")

# Sweep에서 전달받은 파라미터 가져오기
sweep_config = wandb.config

# Config 파일 로드
config_file = 'configs/config.py'
cfg = Config.fromfile(config_file)

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/smoke-test'

# Sweep 파라미터 적용 (범위 동적으로 조정)
if hasattr(sweep_config, 'lr'):
    cfg.optimizer.lr = sweep_config.lr  # 학습률 조정
if hasattr(sweep_config, 'weight_decay'):
    cfg.optimizer.weight_decay = sweep_config.weight_decay  # Weight Decay 조정
if hasattr(sweep_config, 'batch_size'):
    cfg.data.samples_per_gpu = sweep_config.batch_size  # 배치 크기 조정
if hasattr(sweep_config, 'num_epochs'):
    cfg.runner.max_epochs = sweep_config.num_epochs  # 에포크 수 조정

# WandB Config 확인 로그
wandb.config.update(
    {
        "optimizer.lr": cfg.optimizer.lr,
        "optimizer.weight_decay": cfg.optimizer.weight_decay,
        "data.samples_per_gpu": cfg.data.samples_per_gpu,
        "runner.max_epochs": cfg.runner.max_epochs,
    }
)

# 모델 초기화
model = init_segmentor(cfg)

# 학습 시작
train_segmentor(model, cfg)
