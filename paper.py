# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import dlib
import os
import pandas as pd
from base import get_all_images
from pre_processing import PreProcessing
from feature_extractor import FeatureExtractor
# Mở rộng vùng ảnh
def expand_region(x1, y1, w1, h1, scale=0.2):
        """
        Mở rộng vùng ảnh bằng cách thêm padding.
        
        :param x1: Tọa độ x của hình chữ nhật.
        :param y1: Tọa độ y của hình chữ nhật.
        :param w1: Chiều rộng của hình chữ nhật.
        :param h1: Chiều cao của hình chữ nhật.
        :param scale: Tỷ lệ mở rộng.
        :return: Tọa độ mới của hình chữ nhật đã mở rộng.
        """
        # Tính toán kích thước mở rộng
        dx = int(w1 * scale)
        dy = int(h1 * scale)
        x1_expanded = max(x1 - dx, 0)
        y1_expanded = max(y1 - dy, 0)
        w1_expanded = w1 + 2 * dx
        h1_expanded = h1 + 2 * dy
        return (x1_expanded, y1_expanded, w1_expanded, h1_expanded)

def extract_feature(path_img):
    # Load the landmarks model
    predictor_landmarks = dlib.shape_predictor(r"model\landmarks\shape_predictor_68_face_landmarks.dat")  # AAM/ASM landmark model

    # Khởi tạo lớp tiền xử lý dữ liệu
    pre_processing = PreProcessing()

    # Load the image
    image = cv2.imread(path_img)
    image = pre_processing.apply_gamma_correction(image, gamma = 1.5)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    lbp_feature = []
    hog_feature = []
 
    x, y, w, h = 0, 0, 255, 255

    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

    landmarks = predictor_landmarks(gray_image, rect)
    # Trích xuất vùng mắt trái
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    (x1, y1, w1, h1) = cv2.boundingRect(left_eye)
    
    left_eye_region = gray_image[y1:y1 + h1, x1:x1 + w1]
    # Trích xuất vùng mắt phải
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    (x2, y2, w2, h2) = cv2.boundingRect(right_eye)
    right_eye_region = gray_image[y2:y2 + h2, x2:x2 + w2]
    
    # Trích xuất vùng mũi
    nose = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)])
    (x3, y3, w3, h3) = cv2.boundingRect(nose)
    nose_region = gray_image[y3:y3 + h3, x3:x3 + w3]
    
    # Trích xuất vùng miệng
    mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
    (x4, y4, w4, h4) = cv2.boundingRect(mouth)
    mouth_region = gray_image[y4:y4 + h4, x4:x4 + w4]
    
    fe = FeatureExtractor()
    # Trích xuất đặc trưng LBP
    lbp_features_left_eye = fe.LBP(left_eye_region)
    lbp_features_right_eye = fe.LBP(right_eye_region)
    lbp_features_nose_region = fe.LBP(nose_region)
    lbp_features_mouth_region = fe.LBP(mouth_region)

    # Trích xuất đặc trưng HOG
    hog_features_left_eye = fe.extract_hog_features(left_eye_region)
    hog_features_right_eye = fe.extract_hog_features(right_eye_region)
    hog_features_nose_region = fe.extract_hog_features(nose_region)
    hog_features_mouth_region = fe.extract_hog_features(mouth_region)

    # LBP feature
    lbp_feature = np.hstack([lbp_features_left_eye, lbp_features_right_eye, lbp_features_nose_region, lbp_features_mouth_region])
    
    # HOG feature
    hog_feature = np.hstack([hog_features_left_eye, hog_features_right_eye, hog_features_nose_region, hog_features_mouth_region])

    return lbp_feature, hog_feature 

def extract_features(image_list):
    data_lbp = []
    for img in image_list:
        img_path, img_name = os.path.split(img)
        UN = img_name.split("_")[0]

        lbp_fe, hog_fe = extract_feature(img)

        data_lbp.append([UN] + list(lbp_fe))
    return data_lbp

def load_data(positive_dir, negative_dir, split_ratio=0.7):
    positive_images = get_all_images(positive_dir)
    negative_images = get_all_images(negative_dir)

    n_pos_train = int(len(positive_images) * split_ratio)
    n_neg_train = int(len(negative_images) * split_ratio)

    train = np.append(positive_images[:n_pos_train], negative_images[:n_neg_train])
    test = np.append(positive_images[n_pos_train:], negative_images[n_neg_train:])
    
    return train, test

def save_to_csv(data, filename, include_label=True):
    if include_label:
        headers = ["label"] + [f"pixel{i}" for i in range(len(data[0]) - 1)]
    else:
        headers = [f"pixel{i}" for i in range(len(data[0]))]

    df = pd.DataFrame(data, columns=headers)
    df.to_csv(filename, index=False)

# Đường dẫn đến thư mục
positive_dir = r"C:\Users\sang1\OneDrive - The University of Technology\Desktop\DATABASE\databaseV2\emotion\00"
negative_dir = r"C:\Users\sang1\OneDrive - The University of Technology\Desktop\DATABASE\databaseV2\emotion\01"

# Load và chia dữ liệu
train_images, test_images = load_data(positive_dir, negative_dir)

# Trích xuất đặc trưng
train_features = extract_features(train_images)
test_features = extract_features(test_images)

# Lưu train.csv
save_to_csv(train_features, "train.csv")

# Lưu test.csv (chỉ chứa dữ liệu, không có nhãn)
save_to_csv([sample[1:] for sample in test_features], "test.csv", include_label=False)

# Lưu submission.csv (chỉ chứa nhãn)
submission_labels = [[sample[0]] for sample in test_features]
save_to_csv(submission_labels, "submission.csv")