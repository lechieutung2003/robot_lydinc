import cv2
import numpy as np
import os
# from detect import FaceDetector
from joblib import load
from deepface import DeepFace

# Đường dẫn cơ bản
model_dir = "D:\\Artificial Intelligence\\git clone\\robot_lydinc\\model"

# Tạo đường dẫn cho từng mô hình
modelNN_path = os.path.join(model_dir, "NN", "modelNN_Facenet512.joblib")
modelSVM_path = os.path.join(model_dir, "SVM", "modelSVM_Facenet512.joblib")
modelLR_path = os.path.join(model_dir, "LR", "LRv1.joblib")

# Tải các mô hình
modelNN = load(modelNN_path)
modelSVM = load(modelSVM_path)
modelLR = load(modelLR_path)

# Load InsightFace model
# detector = FaceDetector()
backend = "retinaface"
model_name = "Facenet512"
alignment_modes = [True]

cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_path = "temp_frame.jpg"
    cv2.imwrite(img_path, frame)

    detected_faces = DeepFace.represent(
        img_path=img_path,
        detector_backend=backend,
        align=alignment_modes[0],
        enforce_detection=False,
    )

    # for result in detected_faces:
    #     # Unpack the frame and the list of faces
    #     frame, faces = result

    for face in detected_faces:
        # box = face_objs[0]
        x = face["facial_area"]["x"]
        y = face["facial_area"]["y"]
        w = face["facial_area"]["w"]
        h = face["facial_area"]["h"]

        face_img = frame[y : y + h, x : x + w]

        face_objs = DeepFace.represent(
            img_path=face_img,
            # detector_backend=backend,
            align=alignment_modes[0],
            model_name=model_name,
            enforce_detection=False,
        )
        print("face_objs", face_objs)
        embedding = face_objs[0].get("embedding", None)
        embedding = np.array(embedding).reshape(1, -1)

        # Get predictions from all three models
        pred1 = modelNN.predict(embedding)
        pred2 = modelSVM.predict(embedding)
        pred3 = modelLR.predict(embedding)

        # Combine predictions using max voting
        predictions = [pred1[0], pred2[0], pred3[0]]
        final_prediction = max(set(predictions), key=predictions.count)
        print("predictions", predictions)
        print("final_prediction", final_prediction)

        predicted_label = final_prediction

        # Get max probability (for demonstration purposes, using NN model's probability)
        max_probability_NN = np.max(modelNN.predict_proba(embedding))
        max_probability_SVM = np.max(modelSVM.predict_proba(embedding))
        max_probability_LR = np.max(modelLR.predict_proba(embedding))
        max_probability = (max_probability_NN + max_probability_SVM + max_probability_LR) / 3
        print("max_probability", max_probability)

        if max_probability > 0.8:
            predicted_label = f"Person {predicted_label}"
        else:
            predicted_label = "Unknown"

        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            str(predicted_label),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Display the resulting frame
    cv2.imshow("Face Recogniton", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
