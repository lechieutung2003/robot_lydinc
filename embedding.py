import os 
import pandas as pd
import cv2
import pandas as pd
import numpy as np
from detect import FaceDetector
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern
from scipy.fftpack import dct
from skimage.transform import resize
from skimage.io import imread, imshow
from skimage import data, exposure

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

    def process_files(self, fe):
        walk = os.walk(self.root)
        for path, _, files in walk: 
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(path, file)
                    if "train" in img_path:
                        fe(img_path, file, self.train)
                    if "test" in img_path:
                        if "WM" in img_path:
                            fe(img_path, file, self.testWM)
                        if "MM" in img_path:
                            fe(img_path, file, self.testMM)
                        if "HM" in img_path:
                            fe(img_path, file, self.testHM)

    def fe_restAPI(self, img_path, img_name, data):
        label = img_name.split("_")[0]
        
        face_detector = FaceDetector()
        image = cv2.imread(img_path)

        detected_faces = face_detector.detect(image)
        
        for result in detected_faces:
            # Unpack the frame and the list of faces
            frame, faces = result
            
            for face in faces:
                data.append([label] + face.embedding)
        
    def fe_hog(self, img_path, img_name, data):
        label = img_name.split("_")[0]
        img = imread(img_path)
        img = cv2.resize(img, (64,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features = np.array(features)
        dt = [label]
        for feature in features:
            dt.append(feature)
        data.append(dt)

    def pca_reduction(self, n_components=0.95):
        # Kiểm tra xem dữ liệu đã được xử lý chưa
        if not self.train or not self.testWM or not self.testMM or not self.testHM:
            raise ValueError("Dữ liệu không đầy đủ. Không thể giảm chiều")
        
        pca = PCA(n_components=n_components)

        # Tách nhãn và dữ liệu từ self.train
        train_labels = [item[0] for item in self.train]
        train_data = [item[1:] for item in self.train]

        # Tách nhãn và dữ liệu từ self.testWM, self.testMM, self.testHM
        testWM_labels = [item[0] for item in self.testWM]
        testWM_data = [item[1:] for item in self.testWM]

        testMM_labels = [item[0] for item in self.testMM]
        testMM_data = [item[1:] for item in self.testMM]

        testHM_labels = [item[0] for item in self.testHM]
        testHM_data = [item[1:] for item in self.testHM]

        # Áp dụng PCA vào dữ liệu
        train_data_pca = pca.fit_transform(train_data)
        testWM_data_pca = pca.transform(testWM_data)
        testMM_data_pca = pca.transform(testMM_data)
        testHM_data_pca = pca.transform(testHM_data)

        # Gắn lại nhãn vào dữ liệu đã giảm chiều
        self.train = [[label] + list(features) for label, features in zip(train_labels, train_data_pca)]
        self.testWM = [[label] + list(features) for label, features in zip(testWM_labels, testWM_data_pca)]
        self.testMM = [[label] + list(features) for label, features in zip(testMM_labels, testMM_data_pca)]
        self.testHM = [[label] + list(features) for label, features in zip(testHM_labels, testHM_data_pca)]

    def fe_lbp(self, img_path, img_name, data, P=8, R=1):
        """
        Compute the Local Binary Patterns (LBP) of an image.
        
        Parameters:
        - image: Input grayscale image (2D array)
        - P: Number of circularly symmetric neighbor set points (default is 8)
        - R: Radius of circle (default is 1)
        
        Returns:
        - lbp_image: LBP image
        - lbp_hist: Histogram of LBP features
        """
        label = img_name.split("_")[0]
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        lbp_image = local_binary_pattern(img, P, R, method='uniform')
        
        # Compute the histogram of LBP features
        (hist, _) = np.histogram(lbp_image.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        
        # Normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-6)
        
        features = np.array(hist)
        dt = [label]
        for feature in features:
            dt.append(feature)
        data.append(dt)
        
    def fe_pca(img_path, img_name, data, n_components=0.95):
        """
        Extract features using PCA.

        Parameters:
        - img_path: Path to the image file.
        - img_name: Name of the image file.
        - data: List to store the extracted features.
        - n_components: Number of components for PCA. Can be a float (variance ratio) or an integer.
        """
        label = img_name.split("_")[0]
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.flatten()  # Chuyển đổi ảnh 2D thành vector 1D

        pca = PCA(n_components=n_components)
        features = pca.fit_transform([img])[0]  # Lấy đặc trưng sau khi áp dụng PCA

        dt = [label] + list(features)
        data.append(dt)

    def fe_dct(img_path, img_name, data):
        """
        Extract features using Discrete Cosine Transform (DCT).

        Parameters:
        - img_path: Path to the image file.
        - img_name: Name of the image file.
        - data: List to store the extracted features.
        """
        label = img_name.split("_")[0]
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply DCT and flatten the resulting matrix
        dct_features = dct(dct(img.T, norm='ortho').T, norm='ortho').flatten()
        
        dt = [label] + list(dct_features)
        data.append(dt)
    
    def fe_pca_dct(img_path, img_name, data, n_components=0.95):
        """
        Extract features using a combination of DCT and PCA.

        Parameters:
        - img_path: Path to the image file.
        - img_name: Name of the image file.
        - data: List to store the extracted features.
        - n_components: Number of components for PCA. Can be a float (variance ratio) or an integer.
        """
        label = img_name.split("_")[0]
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply DCT
        dct_features = dct(dct(img.T, norm='ortho').T, norm='ortho').flatten()

        # Apply PCA on DCT features
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform([dct_features])[0]
        
        dt = [label] + list(reduced_features)
        data.append(dt)

    def save(self):
        # create train.csv
        header_train = []
        header_train.append(f"label")
        for i in range(len(self.train[0]) - 1):
            header_train.append(f"pixel{i}")
        df_train = pd.DataFrame(self.train, columns=header_train)
        df_train.to_csv("train.csv", index=False)
        
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

        df_submissionWM.to_csv("submissionWM.csv", index=False)
        df_submissionMM.to_csv("submissionMM.csv", index=False)
        df_submissionHM.to_csv("submissionHM.csv", index=False)
       
        # create testWM.csv, testMM.csv, testHM.csv
        header_test = [] 
        for i in range(len(self.testWM[0])):
            header_test.append(f"pixel{i}")
        df_test = pd.DataFrame(self.testWM, columns=header_test)
        df_test.to_csv("testWM.csv", index=False)

        header_test = []
        for i in range(len(self.testMM[0])):
            header_test.append(f"pixel{i}")
        df_test = pd.DataFrame(self.testMM, columns=header_test)
        df_test.to_csv("testMM.csv", index=False)

        header_test = []
        for i in range(len(self.testHM[0])):
            header_test.append(f"pixel{i}")
        df_test = pd.DataFrame(self.testHM, columns=header_test)
        df_test.to_csv("testHM.csv", index=False)



path = r"C:\Users\sang1\OneDrive - The University of Technology\Desktop\DATABASE\databaseV2"

walk = os.walk(path)
embedding = Embedding(path)
embedding.process_files(embedding.fe_hog)
embedding.pca_reduction(512)
embedding.save()
