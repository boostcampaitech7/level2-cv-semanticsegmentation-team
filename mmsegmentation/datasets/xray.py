from typing import List
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class XRayDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
                'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
                'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
                'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
                'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
                'Triquetrum', 'Pisiform', 'Radius', 'Ulna'),
        palette=[[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228],
                [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30],
                [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42],
                [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157],
                [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240],
                [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176]],
        image_size=(2048,2048))

    def __init__(self, image_files, label_files, **kwargs):
        self.image_files = image_files
        self.label_files = label_files  # Optional for test set without labels
        super().__init__(img_suffix='.png', seg_map_suffix='.json', **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        
        data_list = []

        for idx, (img) in enumerate(self.image_files):
            data_info = dict(img_path = img)
            if self.label_files is not None:
                data_info['seg_map_path'] = self.label_files[idx]
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['image_size'] = self._metainfo.get('image_size', (2048, 2048))
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list