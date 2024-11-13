import os
import yaml

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.cfg = yaml.safe_load(file)
        
        # WandB 설정
        self.wandb_group = self.cfg['wandb_group']
        self.wandb_name = self.cfg['wandb_name']

        # KFold 설정
        self.kfold_n = self.cfg['kfold_n']

        # 모델 관련 설정
        self.pretrained = self.cfg['pretrained']
        self.pretrained_dir = self.cfg['pretrained_dir']
        self.device = self.cfg['device']

        # 하이퍼파라미터
        self.batch_size = self.cfg['batch_size']
        self.batch_size_valid = self.cfg['batch_size_valid']
        self.lr = self.cfg['learning_rate']
        self.random_seed = self.cfg['random_seed']
        self.num_epochs = self.cfg['num_epochs']
        self.val_every = self.cfg['val_every']

        # 데이터 및 경로 설정
        self.data_root = self.cfg['data_root']
        self.image_root = self.cfg['image_root']
        self.label_root = self.cfg['label_root']
        self.test_image_root = self.cfg['test_image_root']
        self.saved_dir = os.path.join(self.cfg['saved_dir'], self.wandb_name)

        if not os.path.exists(self.saved_dir):
            os.makedirs(self.saved_dir)

        # 모델 체크포인트 경로
        self.checkpoint_path = self.pretrained_dir if self.pretrained else os.path.join(self.saved_dir, "best_model.pth")

        # 클래스 및 팔레트 설정
        self.classes = self.cfg['classes']
        self.class2ind = {v: i for i, v in enumerate(self.classes)}
        self.ind2class = {i: v for i, v in enumerate(self.classes)}
        self.palette = self.cfg['palette']

# 설정 불러오기
config = Config()
