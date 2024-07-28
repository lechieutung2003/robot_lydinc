import cv2
import numpy as np
import os
from detect import FaceDetector
from utility import PathManager, FaceData
from joblib import load

# Load models
modelNN = load(".\\model\\model_adam_relu_3_64.joblib")
modelSVM = load(".\\model\\modelSVM.joblib")
detector = FaceDetector()

# Đường dẫn cơ sở để lưu ảnh
base_save_path = "F:\LYDINC\Data\complete split"

def save_face(image_path, predicted_label, face_bbox):
    # Tạo đường dẫn lưu dựa trên nhãn dự đoán
    if predicted_label == "unknown":
        save_path = os.path.join(base_save_path, "unknown", os.path.basename(image_path))
    else:
        save_path = os.path.join(base_save_path, str(predicted_label), os.path.basename(image_path))
    
    # Đảm bảo thư mục tồn tại
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    # Đọc ảnh gốc
    frame = cv2.imread(image_path)
    # Cắt khuôn mặt dựa trên bounding box
    x, y, w, h = face_bbox
    face_image = frame[y:h, x:w]
    
    # Kiểm tra xem face_image có rỗng không
    if face_image.size == 0:
        print(f"Không thể cắt khuôn mặt từ {image_path} với bbox {face_bbox}. Bỏ qua.")
        return
    
    face_image = cv2.resize(face_image, (256, 256))
    
    # Lưu ảnh khuôn mặt
    cv2.imwrite(save_path, face_image)

# Cập nhật hàm process_image để trả về cả nhãn và bounding box
def process_image(image_path):
    frame = cv2.imread(image_path)
    detection_results = detector.detect(frame)
    
    if detection_results:
        _, detected_faces = detection_results[-1]  # Get the last item's detected faces
                
        for index, face_data in enumerate(detected_faces):
            x, y, w, h = face_data.bbox
            embedding = np.array(face_data.embedding).reshape(1, -1)

            # Get predictions from both models
            predictions1 = modelNN.predict_proba(embedding)
            predictions2 = modelSVM.predict_proba(embedding)

            # Use max voting
            final_predictions = np.argmax(predictions1 + predictions2, axis=1)
            predicted_label = modelNN.classes_[final_predictions[0]]

            # Get max probability
            max_probability = np.max(predictions1 + predictions2)

            if max_probability > 0.7:
                predicted_label = f"Person {predicted_label}"
            else:
                predicted_label = "Unknown"
            
            # Trả về nhãn và bounding box
            return predicted_label, face_data.bbox

# Cập nhật hàm process_all_images để sử dụng save_face với bounding box
def process_all_images(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            result = process_image(image_path)
            if result is not None:
                predicted_label, face_bbox = result
                save_face(image_path, predicted_label, face_bbox)
            else:
                save_face(image_path, "unknown", None)

# Đường dẫn tới thư mục chứa ảnh
directory_path = "F:\LYDINC\Data origin\image"
process_all_images(directory_path)