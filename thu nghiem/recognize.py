import cv2
import numpy as np
# from detect import FaceDetector
from joblib import load
from deepface import DeepFace

modelNN = load("F:\\LYDINC\\AI\\Hoan\\deepface\\deepface\\model\\model_adam_relu_3_64.joblib")
modelSVM = load("F:\\LYDINC\\AI\\Hoan\\deepface\\deepface\\model\\modelSVM.joblib")

# Load InsightFace model
# detector = FaceDetector()
backend = "yolov8"
model_name = "Facenet"
alignment_modes = [True]

cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    img_path = 'temp_frame.jpg'
    cv2.imwrite(img_path, frame)
        
    detected_faces = DeepFace.represent(
            img_path=img_path,
            detector_backend=backend,
            align=alignment_modes[0],
            enforce_detection=False
        )
        
    # for result in detected_faces:
    #     # Unpack the frame and the list of faces
    #     frame, faces = result
        
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

