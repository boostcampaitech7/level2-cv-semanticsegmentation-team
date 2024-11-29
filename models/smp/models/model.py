import segmentation_models_pytorch as smp
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, model_name, encoder, encoder_weights, in_channels, classes):
        super().__init__()
        
        self.model = getattr(smp, model_name)(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    
    def forward(self, x):
        return {'out': self.model(x)}

ARCHITECTURES = {
    'Unet': 'Unet',
    'UnetPlusPlus': 'UnetPlusPlus',
    'MAnet': 'MAnet',
    'Linknet': 'Linknet',
    'FPN': 'FPN',
    'PSPNet': 'PSPNet',
    'DeepLabV3': 'DeepLabV3',
    'DeepLabV3Plus': 'DeepLabV3Plus',
    'PAN': 'PAN',
    'UperNet': 'UPerNet'
}

def get_model(config):
    if config.MODEL_NAME not in ARCHITECTURES:
        raise ValueError(f"Model {config.MODEL_NAME} not supported. Available models: {list(ARCHITECTURES.keys())}")
    
    model = SegmentationModel(
        model_name=ARCHITECTURES[config.MODEL_NAME],
        encoder=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=len(config.CLASSES),
    )
    
    return model