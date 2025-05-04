class Config:
    CLASS_NUM = 16
    TEST_RATIO = 0.90
    PATCH_SIZE = 25
    PCA_COMPONENTS = 30
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 0.001
    # 定义数据路径
    DATA_PATH = {
        'Indian_pines': {
            'data': 'dataset/Indian_pines_corrected.mat',
            'gt': 'dataset/Indian_pines_gt.mat'
        },
        'PaviaU': {
            'data': 'dataset/PaviaU.mat',
            'gt': 'dataset/PaviaU_gt.mat'
        },
        'Salinas': {
            'data': 'dataset/Salinas_corrected.mat',
            'gt': 'dataset/Salinas_gt.mat'
        }
    }
