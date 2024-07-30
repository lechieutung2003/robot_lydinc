import cv2
import numpy as np
# from detect import FaceDetector
from joblib import load
from deepface import DeepFace
import os

modelNN = load("D:\\Artificial Intelligence\\git clone\\robot_lydinc\\model\\NN\\modelNN_Facenet_adam_relu_3_64.joblib")
modelSVM = load("D:\\Artificial Intelligence\\git clone\\robot_lydinc\\model\\SVM\\modelSVM_Facenet.joblib")

# Load InsightFace model
# detector = FaceDetector()
backend = "yolov8"
model_name = "Facenet"
alignment_modes = [True]
img_path = "D:\\Artificial Intelligence\\git clone\\robot_lydinc\\thu nghiem\\img_test"
output_dir = "D:\\Artificial Intelligence\\git clone\\robot_lydinc\\thu nghiem\\output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for image in os.listdir(img_path):
    image_path = os.path.join(img_path, image)
    img = cv2.imread(image_path)
            
    detected_faces = DeepFace.represent(
            img_path=image_path,
            detector_backend=backend,
            align=alignment_modes[0],
            enforce_detection=False
        )
            
    for face in detected_faces:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            embedding = DeepFace.represent(
                img_path=face,
                # detector_backend=backend,
                align=alignment_modes[0],
                model_name=model_name,
                enforce_detection=False,
            )
            # embedding = np.array(embedding).reshape(1, -1)

            # Get predictions from both models
            predictions1 = modelNN.predict_proba(embedding)
            predictions2 = modelSVM.predict_proba(embedding)

            # Use max voting
            final_predictions = np.argmax(predictions1 + predictions2, axis=1)
            predicted_label = modelNN.classes_[final_predictions[0]]

            # Get max probability
            max_probability = np.max(predictions1 + predictions2)

            if max_probability > 0.8:
                predicted_label = f"Person {predicted_label}"
            else:
                predicted_label = "Unknown"
                
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(
                img,
                str(predicted_label),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    output_path = os.path.join(output_dir)

    # Display the resulting frame
    cv2.imshow("Face Recogniton", img)
    cv2.save(output_path, img)

