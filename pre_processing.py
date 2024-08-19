from __future__ import print_function
import numpy as np
import cv2

class PreProcessing:
    def __init__(self):
        pass
    # Hàm áp dụng gamma correction cho tiền xử lý
    def apply_gamma_correction(self, image, gamma):
        """
        Áp dụng hiệu chỉnh gamma cho ảnh

        Args:
            image: Ảnh đầu vào
            gamma: Giá trị gamma, mặc định là 2.2
        """
        # Kiểm tra đầu vào
        if image is None:
            raise ValueError("Ảnh đầu vào không hợp lệ")
        if gamma <= 0:
            raise ValueError("Giá trị gamma phải lớn hơn 0")

        # Build a lookup table mapping pixel values [0, 255] to their gamma-corrected values
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")

        # Apply the gamma correction using the lookup table
        corrected_image = cv2.LUT(image, table)
        
        return corrected_image
