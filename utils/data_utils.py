import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.io as sio


def apply_pca(X, n_components=30):
    """Apply PCA to hyperspectral data."""
    H, W, C = X.shape
    X = X.reshape(-1, C)
    pca = PCA(n_components=n_components, whiten=True)
    X = pca.fit_transform(X)
    return X.reshape(H, W, n_components)


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def pad_with_zeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def create_image_cubes(X, y, window_size=5, remove_zero_labels=True):
    # 给 X 做 padding
    margin = int((window_size - 1) / 2)
    zeroPaddedX = pad_with_zeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if remove_zero_labels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


def load_data(config):
    X = sio.loadmat(config.DATA_PATH['Indian_pines']['data'])['indian_pines_corrected']
    y = sio.loadmat(config.DATA_PATH['Indian_pines']['gt'])['indian_pines_gt']
    return X, y
