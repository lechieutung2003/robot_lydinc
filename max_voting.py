import cv2
import numpy as np
from detect import FaceDetector
from joblib import load
from scipy.stats import mode


modelLR = load(".\\model\\databaseV5\\modelLogisticRegression_best.joblib")
modelNN = load(".\\model\\databaseV5\\model_adam_relu_3_64.joblib")
modelSVM = load(".\\model\\databaseV5\\modelSVM.joblib")

# Load InsightFace model
detector = FaceDetector()

cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    detected_faces = detector.detect(frame)
        
    for result in detected_faces:
        # Unpack the frame and the list of faces
        frame, faces = result
        
        for face in faces:
            x, y, w, h = face.bbox
            embedding = face.embedding
            embedding = np.array(embedding).reshape(1, -1)

            # Get predictions from both models
            predictionLR = modelLR.predict(embedding)
            predictionSVM = modelSVM.predict(embedding)
            predictionMLP = modelNN.predict(embedding)

            # Use max voting
            predictions = np.array([predictionLR, predictionSVM, predictionMLP])
            final_prediction = mode(predictions, axis=0).mode[0][0]
            predicted_label = modelLR.classes_[final_prediction]

            # Get max probability
            # max_probability = np.max(predictions1 + predictions2)

            # if max_probability > 0.8:
            #     predicted_label = f"Person {predicted_label}"
            # else:
            #     predicted_label = "Unknown"
            
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

