import os
import pandas as pd
import cv2
import pandas as pd
import numpy as np
from detect import FaceDetector


face_detector = FaceDetector()


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

    def process_files(self):
        walk = os.walk(self.root)
        for path, _, files in walk:
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(path, file)
                    if "train" in img_path:
                        self.feature_extraction(img_path, file, self.train)
                    if "test" in img_path:
                        if "WM" in img_path:
                            self.feature_extraction(img_path, file, self.testWM)
                        if "MM" in img_path:
                            self.feature_extraction(img_path, file, self.testMM)
                        if "HM" in img_path:
                            self.feature_extraction(img_path, file, self.testHM)

    def feature_extraction(self, img_path, img_name, data):
        label = img_name.split("_")[0]

        face_detector = FaceDetector()
        image = cv2.imread(img_path)

        detected_faces = face_detector.detect(image)

        for result in detected_faces:
            # Unpack the frame and the list of faces
            frame, faces = result

            for face in faces:
                data.append([label] + face.embedding)

    def save(self):
        # create train.csv
        header_train = []
        header_train.append(f"label")
        for i in range(len(self.train[0]) - 1):
            header_train.append(f"pixel{i}")
        df_train = pd.DataFrame(self.train, columns=header_train)
        df_train.to_csv("train.csv", index=False)

        # # create submission csv files containing only the labels
        # self.submissionWM = [sample[0] for sample in self.testWM]
        # self.submissionMM = [sample[0] for sample in self.testMM]
        # self.submissionHM = [sample[0] for sample in self.testHM]

        # self.testWM = [sample[1:] for sample in self.testWM]
        # self.testMM = [sample[1:] for sample in self.testMM]
        # self.testHM = [sample[1:] for sample in self.testHM]

        # header_submission = []
        # header_submission.append(f"label")
        # df_submissionWM = pd.DataFrame(self.submissionWM, columns=header_submission)
        # df_submissionMM = pd.DataFrame(self.submissionMM, columns=header_submission)
        # df_submissionHM = pd.DataFrame(self.submissionHM, columns=header_submission)

        # df_submissionWM.to_csv("submissionWM.csv", index=False)
        # df_submissionMM.to_csv("submissionMM.csv", index=False)
        # df_submissionHM.to_csv("submissionHM.csv", index=False)

        # # create testWM.csv, testMM.csv, testHM.csv
        # header_test = []
        # for i in range(len(self.testWM[0])):
        #     header_test.append(f"pixel{i}")
        # df_test = pd.DataFrame(self.testWM, columns=header_test)
        # df_test.to_csv("testWM.csv", index=False)

        # header_test = []
        # for i in range(len(self.testMM[0])):
        #     header_test.append(f"pixel{i}")
        # df_test = pd.DataFrame(self.testMM, columns=header_test)
        # df_test.to_csv("testMM.csv", index=False)

        # header_test = []
        # for i in range(len(self.testHM[0])):
        #     header_test.append(f"pixel{i}")
        # df_test = pd.DataFrame(self.testHM, columns=header_test)
        # df_test.to_csv("testHM.csv", index=False)


path = "E:\OneDrive - The University of Technology\DATABASE - Tep cua Tran Van Dinh Sang\databaseV4"

walk = os.walk(path)
embeddings = Embedding(path)
embeddings.process_files()
embeddings.save()
