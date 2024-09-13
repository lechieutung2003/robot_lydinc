import cv2
import numpy as np
from detect import FaceDetector
from joblib import load

modelNN = load(".\\model\\model_adam_relu_3_64.joblib")
modelSVM = load(".\\model\\modelSVM.joblib")

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
            predictions1 = modelNN.predict_proba(embedding)
            # print("predictions1", predictions1)
            predictions2 = modelSVM.predict_proba(embedding)
            # print("predictions2", predictions2)

            # Use max voting
            final_predictions = np.argmax(predictions1 + predictions2, axis=1)
            # print("final_predictions", final_predictions)
            predicted_label = modelNN.classes_[final_predictions[0]]
            # print("predicted_label", predicted_label)

            # Get max probability
            max_probability = np.max(predictions1 + predictions2)
            # print("max_probability", max_probability)

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
