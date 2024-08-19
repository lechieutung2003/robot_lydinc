# import the necessary packages
from __future__ import print_function
import numpy as np
from skimage.feature import hog, local_binary_pattern

class FeatureExtractor:
    def __init__(self):
        pass

    # Hàm trích xuất đặc trưng HOG từ ảnh ROI
    def HOG(self, image):
        """
        Parameter
        ---------
        self: ảnh dạng numpy

        Return
        ------
        hog_features: đặc trưng được trích xuất bằng hog
        """
        # Tính toán HOG
        hog_features = hog(image, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=False, feature_vector=True)
        return hog_features

    # Hàm trích xuất đặc trưng LBP từ ảnh ROI
    def LBP(self, image, n_points=8, radius=1, method='uniform'):
        # Tính toán LBP
        lbp = local_binary_pattern(image, P=n_points, R=radius, method=method)
        
        # Tạo histogram của LBP với 26 bins (tương ứng cho phương pháp 'uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        
        # Chuẩn hóa histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist
    