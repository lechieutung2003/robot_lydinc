from typing import Union
import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace

# Đảm bảo rằng bạn đã định nghĩa hàm detectFace ở đây

# Đường dẫn đến hình ảnh
img_path = "D:\\Artificial Intelligence\\git clone\\deepface\\IMG_9792.jpg"

# Gọi hàm detectFace để phát hiện khuôn mặt
detected_face = DeepFace.detectFace(
    img_path=img_path,
    target_size=(224, 224),
    detector_backend="opencv",
    enforce_detection=True,
    align=True
)

# Kiểm tra xem có khuôn mặt được phát hiện hay không
if detected_face is not None:
    # Chuyển đổi hình ảnh từ BGR sang RGB (định dạng mà Pillow sử dụng)
    img_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
    
    # Giả sử img_rgb là mảng numpy bạn muốn chuyển đổi
    # Đảm bảo rằng mảng có kiểu dữ liệu np.uint8 và kích thước hợp lệ
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).astype(np.uint8)  # Chuyển đổi từ float [0, 1] sang uint8 [0, 255]

    # Sau đó, sử dụng Pillow để chuyển đổi mảng thành đối tượng hình ảnh
    img_pil = Image.fromarray(img_rgb)
    img_pil.show()
else:
    print("Không tìm thấy khuôn mặt trong hình ảnh.")