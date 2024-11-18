from mmengine.config import Config
from mmseg.apis import init_segmentor, inference_segmentor
import mmcv

config_file = 'configs/hand_bone.py'
checkpoint_file = 'work_dirs/latest.pth'
cfg = Config.fromfile(config_file)

# 모델 로드
model = init_segmentor(cfg, checkpoint_file)

# 테스트
img = 'data/test/images/sample.jpg'
result = inference_segmentor(model, img)
model.show_result(img, result, out_file='result.jpg')
