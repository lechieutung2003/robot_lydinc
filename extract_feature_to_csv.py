import os
import pandas as pd
import cv2
import pandas as pd
import numpy as np
from detect import FaceDetector
from utility import PathManager

face_detector = FaceDetector()

pathManager = PathManager()


class Embedding:
    def __init__(self, images_dir, csv_dir):

        self.images_dir = images_dir
        self.csv_dir = csv_dir
        self.train = []
        self.testWM = []
        self.testMM = []
        self.testHM = []
        self.submissionWM = []
        self.submissionMM = []
        self.submissionHM = []

    def process_files(self):
        images_path = pathManager.get_all_images(self.images_dir)

        for img in images_path:
            img_name = os.path.split(img)[-1].split(".")[0]

            if "train" in img:
                self.feature_extraction(img, img_name, self.train)
            if "test" in img:
                if "WM" in img:
                    self.feature_extraction(img, img_name, self.testWM)
                if "MM" in img:
                    self.feature_extraction(img, img_name, self.testMM)
                if "HM" in img:
                    self.feature_extraction(img, img_name, self.testHM)

    def feature_extraction(self, img_path, img_name, data):
        if "identify" in img_path:
            label = img_name.split("_")[0]
        elif "emotion" in img_path:
            label = img_name.split("_")[2]

        face_detector = FaceDetector()
        image = cv2.imread(img_path)

        detected_faces = face_detector.detect(image)

        for result in detected_faces:
            # Unpack the frame and the list of faces
            frame, faces = result

            for face in faces:
                data.append([label] + face.embedding)

    def save(self):
        train_path = os.path.join(self.csv_dir, "train.csv")
        testWM_path = os.path.join(self.csv_dir, "testWM.csv")
        testMM_path = os.path.join(self.csv_dir, "testMM.csv")
        testHM_path = os.path.join(self.csv_dir, "testHM.csv")
        submissionWM_path = os.path.join(self.csv_dir, "submissionWM.csv")
        submissionMM_path = os.path.join(self.csv_dir, "submissionMM.csv")
        submissionHM_path = os.path.join(self.csv_dir, "submissionHM.csv")
        # create train.csv
        if self.train:
            header_train = []
            header_train.append(f"label")
            for i in range(len(self.train[0]) - 1):
                header_train.append(f"pixel{i}")
            df_train = pd.DataFrame(self.train, columns=header_train)
            df_train.to_csv(train_path, index=False)

        # create submission csv files containing only the labels
        if self.testWM:
            self.submissionWM = [sample[0] for sample in self.testWM]
            self.testWM = [sample[1:] for sample in self.testWM]
            header_submission = []
            header_submission.append(f"label")
            df_submissionWM = pd.DataFrame(self.submissionWM, columns=header_submission)
            df_submissionWM.to_csv(submissionWM_path, index=False)

            # create testWM.csv
            header_test = []
            for i in range(len(self.testWM[0])):
                header_test.append(f"pixel{i}")
            df_test = pd.DataFrame(self.testWM, columns=header_test)
            df_test.to_csv(testWM_path, index=False)

        if self.testMM:
            self.submissionMM = [sample[0] for sample in self.testMM]
            self.testMM = [sample[1:] for sample in self.testMM]
            df_submissionMM = pd.DataFrame(self.submissionMM, columns=header_submission)
            df_submissionMM.to_csv(submissionMM_path, index=False)

            # create testWM.csv, testMM.csv, testHM.csv
            header_test = []
            for i in range(len(self.testMM[0])):
                header_test.append(f"pixel{i}")
            df_test = pd.DataFrame(self.testMM, columns=header_test)
            df_test.to_csv(testMM_path, index=False)

        if self.testHM:
            self.submissionHM = [sample[0] for sample in self.testHM]
            self.testHM = [sample[1:] for sample in self.testHM]
            df_submissionHM = pd.DataFrame(self.submissionHM, columns=header_submission)
            df_submissionHM.to_csv(submissionHM_path, index=False)

            # create testWM.csv, testMM.csv, testHM.csv
            header_test = []
            for i in range(len(self.testHM[0])):
                header_test.append(f"pixel{i}")
            df_test = pd.DataFrame(self.testHM, columns=header_test)
            df_test.to_csv(testHM_path, index=False)


images_dir = r"E:\OneDrive - The University of Technology\DATABASE - Tep cua Tran Van Dinh Sang\databaseV4"
emotion_csv = r"csv\databaseV4"
walk = os.walk(
    images_dir,
)
embeddings = Embedding(images_dir, emotion_csv)
embeddings.process_files()
embeddings.save()
