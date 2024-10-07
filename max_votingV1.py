import cv2
import numpy as np
from detect import FaceDetector
from joblib import load
from collections import Counter
import pandas as pd
import os

# Tải các mô hình
modelLR = load(".\\model\\databaseV5\\modelLogisticRegression_best.joblib")
modelNN = load(".\\model\\databaseV5\\modelNN_adam_relu_3_64.joblib")
modelSVM = load(".\\model\\databaseV5\\modelSVM.joblib")

# Khởi tạo FaceDetector
detector = FaceDetector()

video_path = "F:\\LYDINC\\AI\\robot_lydinc\\video\\IMG_9686.MOV"
cap = cv2.VideoCapture(video_path)

# # Tạo thư mục lưu trữ frame nếu chưa tồn tại
# output_dir = "processed_frames"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# frame_count = 0

# Nhóm các bounding box gần nhau
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def group_bounding_boxes(bounding_boxes, distance_threshold):
    groups = []
    used = set()  # Để theo dõi các bounding box đã được nhóm
    for key, bbox in bounding_boxes.items():
        if key in used:
            continue

        group = [key]
        used.add(key)
        center1 = get_center(bbox)

        for other_key, other_bbox in bounding_boxes.items():
            if other_key in used:
                continue
                
            center2 = get_center(other_bbox)
            distance = np.linalg.norm(np.array(center1) - np.array(center2))

            if distance < distance_threshold:
                group.append(other_key)
                used.add(other_key)

        groups.append(group)

    return groups

# Gán nhãn cho các nhóm
def assign_label_to_groups(groups, data):
    labeled_groups = []
    for group in groups:
        # Lấy các nhãn trong nhóm
        labels = [data[key] for key in group]
        label_count = Counter(labels)
            
        # Gán nhãn nếu có nhãn nào xuất hiện trên 5 lần
        assigned_label = None
        for label, count in label_count.items():
            if count > 7:
                assigned_label = label
                break

        labeled_groups.append((group, assigned_label))

    return labeled_groups

# Đọc video
while True:
    frames = []
    
    # Đọc 7 frame
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # if len(frames) < 10:  # Kiểm tra xem có đủ frame không
    #     break

    # Lưu trữ các frame đang xử lý
    # for idx, frame in enumerate(frames):
    #     frame_filename = os.path.join(output_dir, f"frame_{frame_count + idx}.jpg")
    #     cv2.imwrite(frame_filename, frame)
    
    # frame_count += 7
    
    # Lưu trữ thông tin về label và bbox
    bbox_labels = {}  # Để lưu thông tin bbox và label của từng frame

    for frame_idx, frame in enumerate(frames):
        detected_faces = detector.detect(frame)

        for result in detected_faces:
            frame, faces = result
            
            for i, face in enumerate(faces):
                x, y, w, h = face.bbox
                embedding = face.embedding
                embedding = np.array(embedding).reshape(1, -1)

                # Dự đoán từ các mô hình
                predictionLR = modelLR.predict(embedding)[0]
                predictionSVM = modelSVM.predict(embedding)[0]
                predictionMLP = modelNN.predict(embedding)[0]

                # Sử dụng bỏ phiếu tối đa
                predictions = np.array([predictionLR, predictionSVM, predictionMLP])
                final_prediction = Counter(predictions).most_common(1)[0][0]

                # Tạo key cho bbox
                bbox_key = f"{x}_{y}_{w}_{h}"
                
                # Lưu label vào dict với key là bbox
                if bbox_key not in bbox_labels:
                    bbox_labels[bbox_key] = [final_prediction]
                else:
                    bbox_labels[bbox_key].append(final_prediction)

    # Tính toán label cuối cùng cho từng bbox
    final_results = {}
    for bbox_key, labels in bbox_labels.items():
        # Tính toán số lần xuất hiện của từng label
        final_label = Counter(labels).most_common(1)[0][0]
        final_results[bbox_key] = final_label

    # Tạo danh sách các bounding boxes từ final_results
    bounding_boxes = {key: tuple(map(int, key.split('_'))) for key in final_results.keys()}

    # Nhóm các bounding box gần nhau
    distance_threshold = 50  # Thay đổi khoảng cách ngưỡng ở đây nếu cần
    groups = group_bounding_boxes(bounding_boxes, distance_threshold)

    # Gán nhãn cho các nhóm
    labeled_groups = assign_label_to_groups(groups, final_results)

    # Vẽ bounding box và gán nhãn lên frame thứ 7
    frame10 = frames[9].copy()  # Frame thứ 7
    for group, label in labeled_groups:
        print(group, label)
        if group:
            key = group[-1]  # Lấy key của bbox cuối cùng trong nhóm
            print(key)
            bbox = bounding_boxes[key]
            x1, y1, x2, y2 = bbox
            # Vẽ bounding box
            cv2.rectangle(frame10, (x1, y1), (x2, y2), (0, 0, 225), 2)
            # Gán nhãn
            # if label is not None:
            label_text = str(label)
            cv2.putText(frame10, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Hiển thị frame thứ 10
    cv2.imshow("Frame 10", frame10)
    cv2.waitKey(1)  # Nhấn phím bất kỳ để đóng cửa sổ

    # Xóa các bounding box đã xử lý xong
    bounding_boxes.clear()  # Hoặc gán bounding_boxes = {}

    # Nếu cần thiết, cũng có thể xóa hoặc làm mới các group và label nếu chúng không cần lưu lại
    groups.clear()
    labeled_groups.clear()

cap.release()
cv2.destroyAllWindows()

