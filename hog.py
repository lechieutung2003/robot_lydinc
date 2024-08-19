import cv2
from sklearn.decomposition import PCA
import numpy as np

# Đọc hình ảnh
image = cv2.imread(r"C:\Users\sang1\OneDrive - The University of Technology\Desktop\DATABASE\databaseV2\train\00\HM\01_60_00_N_S0_FDR0_IQ3_F1.jpg")

# Chuyển đổi hình ảnh sang màu xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Khởi tạo đối tượng HOGDescriptor
hog = cv2.HOGDescriptor()

# Trích xuất đặc trưng HOG
features = hog.compute(gray_image)
features = np.array(features)
print(features.shape)
# features = features.reshape(-1,1)
# pca = PCA(n_components=100)  # Giảm xuống 100 chiều
# reduced_features = pca.fit_transform(features)
# print(len(features))
