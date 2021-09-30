"""
This file records the directory paths to the different datasets.
You will need to configure it for training the model.

All datasets follows the following format, where fgr and pha points to directory that contains jpg or png.
Inside the directory could be any nested formats, but fgr and pha structure must match. You can add your own
dataset to the list as long as it follows the format. 'fgr' should point to foreground images with RGB channels,
'pha' should point to alpha images with only 1 grey channel.
{
    'YOUR_DATASET': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        }
    }
}
"""

DATA_PATH = {
    'videomatte240k': {
        'train': {
            'fgr': '/home/sp/zhaojl/BackgroundMattingV2-master/data/VideoMatte240K_JPEG_SD/train/fgr',
            'pha': '/home/sp/zhaojl/BackgroundMattingV2-master/data/VideoMatte240K_JPEG_SD/train/pha'
        },
        # 'train': {
        #     'fgr': '/home/sp/zhaojl/BackgroundMattingV2-master/data/mymattingdatasets/train/fgr',
        #     'pha': '/home/sp/zhaojl/BackgroundMattingV2-master/data/mymattingdatasets/train/pha'
        # },
        'valid': {
            'fgr': '/home/sp/zhaojl/BackgroundMattingV2-master/data/VideoMatte240K_JPEG_SD/test/fgr',
            'pha': '/home/sp/zhaojl/BackgroundMattingV2-master/data/VideoMatte240K_JPEG_SD/test/pha'
        }
    },
    'photomatte13k': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        }
    },
    'distinction': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'adobe': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'backgrounds': {
        'train': '/home/sp/zhaojl/BackgroundMattingV2-master/data/Backgrounds/',
        'valid': '/home/sp/zhaojl/BackgroundMattingV2-master/data/Backgrounds/'
    },
}