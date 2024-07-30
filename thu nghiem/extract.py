from deepface import DeepFace
import numpy as np
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import random

backends = [
    "retinaface",
    "yolov8",
]

model_names = [
    # "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    # "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace",
    "GhostFaceNet",
]

alignment_modes = [True, False]
img_path = ""


def extract_features(img_path, backend, model_name):

    alignment_modes = [True, False]

    face_objs = DeepFace.represent(
        img_path=img_path,
        # detector_backend=backend,
        align=alignment_modes[0],
        model_name=model_name,
        enforce_detection=False,
    )
    # print(backend, model_name)

    embedding = face_objs[0].get("embedding", None)
    # Ensure the embedding is returned as a NumPy array
    feature_vector = np.array(embedding) if embedding is not None else None
    return feature_vector


path_train = r"F:\LYDINC\AI\Hoan\deepface\deepface\data\identity\train"
walk_train = os.walk(path_train)

path_test = r"F:\LYDINC\AI\Hoan\deepface\deepface\data\identity\test"
walk_test = os.walk(path_test)
print(walk_test)

base_dir = r"F:\LYDINC\AI\Hoan\deepface\deepface\data\csv"

# Vòng lặp qua mỗi backend
def feature_extract(walk_train, walk_test, base_dir, backend, model_name):
    
        print(f"Extracting features using {backend} and {model_name}")

        output_dir = os.path.join(base_dir, f"{backend}-{model_name}")
        # Kiểm tra xem thư mục đã tồn tại chưa, nếu chưa thì tạo mới
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_train = []
        for path, dirs, files in walk_train:
            
            if len(files) == 0:
                continue

            for file in files:
                img_path = os.path.join(path, file)
                print(img_path)

                emotion = file.split("_")[0]
                emotion = emotion.split(".")[0]

                vec = extract_features(img_path, backend, model_name)

                if vec is not None:  # Kiểm tra vec không phải là None
                    emotion_array = np.array([emotion])  # Chuyển emotion thành mảng
                    vec = np.append(emotion_array, vec)  # Thêm emotion vào đầu vec
                    data_train.append(vec)
                    # print(data_train)
                else:
                    print(f"Cannot extract features for {img_path}")  # Xử lý trường hợp vec là None
            samples_train = int(len(data_train))

        data_test_WM = []
        data_test_MM = []
        data_test_HM = []
        all_files = []
        for path, dirs, files in walk_test:
            if "WM" in path:
                
                if len(files) == 0:
                    continue
                
                for file in files:
                    img_path = os.path.join(path, file)
                    print(img_path)

                    emotion_WM = file.split("_")[0]
                    emotion_WM = emotion_WM.split(".")[0]

                    vec_WM = extract_features(img_path, backend, model_name)

                    if vec_WM is not None:  # Kiểm tra vec không phải là None
                        emotion_array_WM = np.array([emotion_WM])  # Chuyển emotion thành mảng
                        vec_WM = np.append(emotion_array_WM, vec_WM)  # Thêm emotion vào đầu vec
                        data_test_WM.append(vec_WM)
                    else:
                        print(
                            f"Cannot extract features for {img_path}"
                        )  # Xử lý trường hợp vec là None


            if "MM" in path:
                
                if len(files) == 0:
                    continue
                
                for file in files:
                    img_path = os.path.join(path, file)
                    print(img_path)

                    emotion_MM = file.split("_")[0]
                    emotion_MM = emotion_MM.split(".")[0]

                    vec_MM = extract_features(img_path, backend, model_name)

                    if vec_MM is not None:  # Kiểm tra vec không phải là None
                        emotion_array_MM = np.array([emotion_MM])  # Chuyển emotion thành mảng
                        vec_MM = np.append(emotion_array_MM, vec_MM)  # Thêm emotion vào đầu vec
                        data_test_MM.append(vec_MM)
                    else:
                        print(
                            f"Cannot extract features for {img_path}"
                        )  # Xử lý trường hợp vec là None


            if "HM" in path:
                
                if len(files) == 0:
                    continue
                
                for file in files:
                    img_path = os.path.join(path, file)
                    print(img_path)

                    emotion_HM = file.split("_")[0]
                    emotion_HM = emotion_HM.split(".")[0]

                    vec_HM = extract_features(img_path, backend, model_name)

                    if vec_HM is not None:  # Kiểm tra vec không phải là None
                        emotion_array_HM = np.array([emotion_HM])  # Chuyển emotion thành mảng
                        vec_HM = np.append(emotion_array_HM, vec_HM)  # Thêm emotion vào đầu vec
                        data_test_HM.append(vec_HM)
                    else:
                        print(
                            f"Cannot extract features for {img_path}"
                        )  # Xử lý trường hợp vec là None


        # samples_train = int(len(data) * 0.6)
        # samples_test = len(data) - samples_train

        # train = data[:samples_train]
        # test = data[samples_train:]
        
        submission_WM = [sample[0] for sample in data_test_WM]
        test_WM = np.array([np.delete(sample, 0) for sample in data_test_WM])
        
        submission_MM = [sample[0] for sample in data_test_MM]
        test_MM = np.array([np.delete(sample, 0) for sample in data_test_MM])
        
        submission_HM = [sample[0] for sample in data_test_HM]
        test_HM = np.array([np.delete(sample, 0) for sample in data_test_HM])
        # print(test[0])

        [header_train, header_test, header_submission] = [[], [], []]
        header_test_WM = []
        header_test_MM = []
        header_test_HM = []
        header_submission_WM = []
        header_submission_MM = []
        header_submission_HM = []

        header_train.append(f"label")
        for i in range(len(data_train[0]) - 1):
            header_train.append(f"pixel{i}")
        df_train = pd.DataFrame(data_train, columns=header_train)
        df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        
        #luu csv cho WM
        # if test_WM and len(test_WM) > 0:
        for i in range(len(test_WM[0])):
            header_test_WM.append(f"pixel{i}")
        df_test_WM = pd.DataFrame(test_WM, columns=header_test_WM)
        df_test_WM.to_csv(os.path.join(output_dir, "test_WM.csv"), index=False)
            
        header_submission_WM.append(f"label")
        df_submission_WM = pd.DataFrame(submission_WM, columns=header_submission_WM)
        df_submission_WM.to_csv(os.path.join(output_dir, "submission_WM.csv"), index=False)
        
        #Luu csv cho MM
        # if test_MM and len(test_MM) > 0:
        for i in range(len(test_MM[0])):
            header_test_MM.append(f"pixel{i}")
        df_test_MM = pd.DataFrame(test_MM, columns=header_test_MM)
        df_test_MM.to_csv(os.path.join(output_dir, "test_MM.csv"), index=False)
            
        header_submission_MM.append(f"label")
        df_submission_MM = pd.DataFrame(submission_MM, columns=header_submission_MM)
        df_submission_MM.to_csv(os.path.join(output_dir, "submission_MM.csv"), index=False)
        
        #Luu csv cho HM
        
        for i in range(len(test_HM[0])):
            header_test_HM.append(f"pixel{i}")
        df_test_HM = pd.DataFrame(test_HM, columns=header_test_HM)
        df_test_HM.to_csv(os.path.join(output_dir, "test_HM.csv"), index=False)
            
        header_submission_HM.append(f"label")
        df_submission_HM = pd.DataFrame(submission_HM, columns=header_submission_HM)
        df_submission_HM.to_csv(os.path.join(output_dir, "submission_HM.csv"), index=False)
        
        
backends = [
    "retinaface",
    "yolov8",
]

model_names = [
    # "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    # "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace",
    "GhostFaceNet",
]        

feature_extract(walk_train, walk_test, base_dir, backends[1], model_names[7])
       