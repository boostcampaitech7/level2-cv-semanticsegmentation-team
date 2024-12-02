# config/config.py

import os
from datetime import datetime

class Config:
    def __init__(self, args=None):
        # Base directory
        self.BASE_DIR = args.base_dir if args else "/data/ephemeral/home/data"
        
        # Data paths
        self.IMAGE_ROOT = os.path.join(self.BASE_DIR, "train/DCM")
        self.LABEL_ROOT = os.path.join(self.BASE_DIR, "train/outputs_json")
        self.TEST_ROOT = os.path.join(self.BASE_DIR, "test/DCM")
        
        # Model
        self.MODEL_NAME = args.model_name if args else "UnetPlusPlus"
        self.ENCODER = args.encoder if args else "efficientnet-b0"
        self.ENCODER_WEIGHTS = args.encoder_weights if args else "imagenet"
        self.IN_CHANNELS = 3
        
        # Training
        self.BATCH_SIZE = args.batch_size if args else 8
        self.LEARNING_RATE = args.lr if args else 1e-4
        self.RANDOM_SEED = args.seed if args else 42
        self.NUM_EPOCHS = args.num_epochs if args else 50  # Changed from args.epochs
        self.NUM_WORKERS = args.num_workers if args else 4
        self.FOLD = args.fold if hasattr(args, 'fold') else None
        
        # Checkpoint
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.EXP_NAME = f"{self.MODEL_NAME}_{self.ENCODER}_fold{self.FOLD}" if self.FOLD is not None else f"{self.MODEL_NAME}_{self.ENCODER}_{current_time}"
        self.SAVED_DIR = os.path.join("experiments", self.EXP_NAME)
        
        # Classes
        self.CLASSES = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
        self.CLASS2IND = {v: i for i, v in enumerate(self.CLASSES)}
        self.IND2CLASS = {v: k for k, v in self.CLASS2IND.items()}