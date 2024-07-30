import os 
import pandas as pd
import cv2
import pandas as pd
import numpy as np
# from detect import FaceDetector
from deepface import DeepFace


# face_detector = FaceDetector()

class Embedding:
    def __init__(self, root):
        self.root = root
        self.train = []
        self.testWM = []
        self.testMM = []
        self.testHM = []
        self.submissionWM = []
        self.submissionMM = []
        self.submissionHM = []
        self.model_names = ["Facenet", "Facenet512", "OpenFace", "DeepID", "Dlib", "ArcFace", "SFace", "GhostFaceNet"]
        self.alignment_modes = [True, False]

    # def process_files(self):
    #     walk = os.walk(self.root)
    #     for path, _, files in walk: 
    #         for file in files:
    #             if file.endswith(".jpg"):
    #                 img_path = os.path.join(path, file)
    #                 if "train" in img_path:
    #                     self.feature_extraction(img_path, file, self.train)
    #                 if "test" in img_path:
    #                     if "WM" in img_path:
    #                         self.feature_extraction(img_path, file, self.testWM)
    #                     if "MM" in img_path:
    #                         self.feature_extraction(img_path, file, self.testMM)
    #                     if "HM" in img_path:
    #                         self.feature_extraction(img_path, file, self.testHM)

    def process_files_DeepFace(self, model_name):
        walk = os.walk(self.root)
        for path, _, files in walk: 
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(path, file)
                    if "train" in img_path:
                        self.extract_features_DeepFace(img_path, file, model_name, self.train)
                    if "test" in img_path:
                        if "WM" in img_path:
                            self.extract_features_DeepFace(img_path, file, model_name, self.testWM)
                        if "MM" in img_path:
                            self.extract_features_DeepFace(img_path, file, model_name, self.testMM)
                        if "HM" in img_path:
                            self.extract_features_DeepFace(img_path, file, model_name, self.testHM)
        

    # def feature_extraction(self, img_path, img_name, data):
    #     label = img_name.split("_")[0]
        
    #     face_detector = FaceDetector()
    #     image = cv2.imread(img_path)

    #     detected_faces = face_detector.detect(image)
        
    #     for result in detected_faces:
    #         # Unpack the frame and the list of faces
    #         frame, faces = result
            
    #         for face in faces:
    #             data.append([label] + face.embedding)
    
    def extract_features_DeepFace(self, img_path, img_name, model_name, data):
        label = img_name.split("_")[0]
        print(img_path)
        face_objs = DeepFace.represent(
            img_path=img_path,
            align=True,
            model_name=model_name,
            enforce_detection=False,
        )
        
        embedding = face_objs[0].get("embedding", None)
        # feature_vector = np.array(embedding) if embedding is not None else None
        print(embedding)
        if embedding is not None:
            data.append([str(label)] + embedding)
        else:
            print(f"Failed to extract features from {img_path}")
        


    # def save(self):
    #     # create train.csv
    #     header_train = []
    #     header_train.append(f"label")
    #     for i in range(len(self.train[0]) - 1):
    #         header_train.append(f"pixel{i}")
    #     df_train = pd.DataFrame(self.train, columns=header_train)
    #     df_train.to_csv("train.csv", index=False)
        
    #     # create submission csv files containing only the labels
    #     self.submissionWM = [sample[0] for sample in self.testWM]
    #     self.submissionMM = [sample[0] for sample in self.testMM]
    #     self.submissionHM = [sample[0] for sample in self.testHM]

    #     self.testWM = [sample[1:] for sample in self.testWM]
    #     self.testMM = [sample[1:] for sample in self.testMM]
    #     self.testHM = [sample[1:] for sample in self.testHM]
        
    #     header_submission = []
    #     header_submission.append(f"label")
    #     df_submissionWM = pd.DataFrame(self.submissionWM, columns=header_submission)
    #     df_submissionMM = pd.DataFrame(self.submissionMM, columns=header_submission)
    #     df_submissionHM = pd.DataFrame(self.submissionHM, columns=header_submission)

    #     df_submissionWM.to_csv("submissionWM.csv", index=False)
    #     df_submissionMM.to_csv("submissionMM.csv", index=False)
    #     df_submissionHM.to_csv("submissionHM.csv", index=False)
       
    #     # create testWM.csv, testMM.csv, testHM.csv
    #     header_test = [] 
    #     for i in range(len(self.testWM[0])):
    #         header_test.append(f"pixel{i}")
    #     df_test = pd.DataFrame(self.testWM, columns=header_test)
    #     df_test.to_csv("testWM.csv", index=False)

    #     header_test = []
    #     for i in range(len(self.testMM[0])):
    #         header_test.append(f"pixel{i}")
    #     df_test = pd.DataFrame(self.testMM, columns=header_test)
    #     df_test.to_csv("testMM.csv", index=False)

    #     header_test = []
    #     for i in range(len(self.testHM[0])):
    #         header_test.append(f"pixel{i}")
    #     df_test = pd.DataFrame(self.testHM, columns=header_test)
    #     df_test.to_csv("testHM.csv", index=False)

    def save_DeepFace(self, output_dir="output"):
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # create train.csv
        header_train = []
        header_train.append(f"label")
        for i in range(len(self.train[0])-1):
            header_train.append(f"pixel{i}")
        df_train = pd.DataFrame(self.train, columns=header_train)
        df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        
        # create submission csv files containing only the labels
        self.submissionWM = [sample[0] for sample in self.testWM]
        self.submissionMM = [sample[0] for sample in self.testMM]
        self.submissionHM = [sample[0] for sample in self.testHM]

        self.testWM = [sample[1:] for sample in self.testWM]
        self.testMM = [sample[1:] for sample in self.testMM]
        self.testHM = [sample[1:] for sample in self.testHM]
        
        header_submission = []
        header_submission.append(f"label")
        df_submissionWM = pd.DataFrame(self.submissionWM, columns=header_submission)
        df_submissionMM = pd.DataFrame(self.submissionMM, columns=header_submission)
        df_submissionHM = pd.DataFrame(self.submissionHM, columns=header_submission)

        df_submissionWM.to_csv(os.path.join(output_dir, "submissionWM.csv"), index=False)
        df_submissionMM.to_csv(os.path.join(output_dir, "submissionMM.csv"), index=False)
        df_submissionHM.to_csv(os.path.join(output_dir, "submissionHM.csv"), index=False)
       
        # create testWM.csv, testMM.csv, testHM.csv
        header_test = [] 
        for i in range(len(self.testWM[0])):
            header_test.append(f"pixel{i}")
        df_test = pd.DataFrame(self.testWM, columns=header_test)
        df_test.to_csv(os.path.join(output_dir, "testWM.csv"), index=False)

        header_test = []
        for i in range(len(self.testMM[0])):
            header_test.append(f"pixel{i}")
        df_test = pd.DataFrame(self.testMM, columns=header_test)
        df_test.to_csv(os.path.join(output_dir, "testMM.csv"), index=False)

        header_test = []
        for i in range(len(self.testHM[0])):
            header_test.append(f"pixel{i}")
        df_test = pd.DataFrame(self.testHM, columns=header_test)
        df_test.to_csv(os.path.join(output_dir, "testHM.csv"), index=False)



path = "D:\\Artificial Intelligence\\git clone\\robot_lydinc\\thu nghiem\\databaseV1"

embeddings = Embedding(path)
# embeddings.process_files()
embeddings.process_files_DeepFace("Facenet512")
# embeddings.save()
embeddings.save_DeepFace()