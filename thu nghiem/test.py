from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

backends = [
  # 'opencv', 
  # 'ssd', 
  # 'dlib', 
  # 'mtcnn', 
  # 'fastmtcnn',
  # 'retinaface', 
  # 'mediapipe',
  'yolov8',
  # 'yunet',
  # 'centerface',
]

model_names = [
    'VGG-Face', 
    'Facenet', 
    'Facenet512', 
    'OpenFace', 
    'DeepFace', 
    'DeepID', 
    'Dlib', 
    'ArcFace', 
    'SFace', 
    'GhostFaceNet',
]

alignment_modes = [True, False]
img_path = "F:\\LYDINC\\AI\\Hoan\\deepface\\deepface\\test image\\v1\\IMG_9792.jpg"
save_path = "F:\\LYDINC\\AI\\Hoan\\deepface\\deepface\\test image\\v1"
log_file_path = save_path + "detection_log.txt"


for backend in backends:
    
        print(f"backend: {backend}")
        
        # Sử dụng OpenCV để đọc hình ảnh
        original_img = cv2.imread(img_path)

        try:
            # face detection and alignment
            face_objs = DeepFace.represent(
              img_path = img_path, 
              detector_backend = backend,
              align = alignment_modes[0],
            #   model_name = model_name,
              enforce_detection=False
            )
            
            detected_faces_count = len(face_objs) if face_objs else 0
            print(f"Detected faces: {detected_faces_count}")

            # Vẽ box bao quanh khuôn mặt nếu có
            if face_objs:
                for face_info in face_objs:
                    facial_area = face_info["facial_area"]
                    x = facial_area['x']
                    y = facial_area['y']
                    w = facial_area['w']
                    h = facial_area['h']
                    cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            filename = f"{backend}.jpg".replace(" ", "_")

            # Tạo đường dẫn đầy đủ để lưu ảnh
            full_path = save_path + filename

            # Lưu ảnh vào đường dẫn mới
            cv2.imwrite(full_path, original_img)
            print(f"Image saved as {full_path}")

            # Mở file để ghi thêm (append) và ghi số lượng khuôn mặt được phát hiện
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"backend: {backend} - detected faces: {detected_faces_count}\n")
        except Exception as e:
            print(f"Error processing with backend: {backend} and model:. Error: {e}")