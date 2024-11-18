import os.path as osp
import argparse
import logging
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import RUNNERS
import wandb

def parse_args():
    """CLI arguments for smoke test."""
    parser = argparse.ArgumentParser(description="Run Smoke Test for MMsegmentation")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory to save logs and models"
    )

    return parser.parse_args()

def run_smoke_test(args):
    """
    Smoke Test: 기본 동작 확인을 위한 빠른 실행.
    
    Args:
        args: CLI arguments containing config path, work_dir, and max_iters.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SmokeTest")

    # Config 파일 로드
    cfg = Config.fromfile(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Smoke Test용 설정 적용
    cfg.evaluation.interval = 1            # 1 에포크마다 평가
    cfg.log_config.interval = 1            # 1 iteration마다 로그 출력

    # Set up working dir to save files and logs.
    if args.work_dir is not None:
        # CLI로 전달된 work_dir 사용
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # config 파일 이름을 기본 work_dir로 사용
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    else:
        # Config 파일에 정의된 work_dir 사용
        cfg.work_dir = cfg.work_dir

    logger.info(f"Work directory: {cfg.work_dir}")

    # build the runner from config
    if 'runner_type' not in cfg:
        # Default runner 사용
        runner = Runner.from_cfg(cfg)
    else:
        # 커스터마이즈된 runner 사용
        runner = RUNNERS.build(cfg)

    # 학습 시작
    logger.info("Starting smoke test training...")
    runner.train()
    logger.info("Smoke test completed successfully.")

if __name__ == "__main__":
    # wandb.login(key="97ae30a3f47da7ce905379af5f6bfc8d2d4dd531")

    # CLI 인자 파싱
    args = parse_args()

    # Smoke Test 실행
    run_smoke_test(args)
