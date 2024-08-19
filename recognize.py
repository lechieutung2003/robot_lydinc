import cv2
import numpy as np
from detect import FaceDetector
from joblib import load
import os

# Load the model train
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

#             predictions = modelSVM.predict(embedding)
#             predicted_label = predictions[0]

            predictions = model.predict_proba(embedding)
            max_probability = np.max(predictions)
            predicted_label = model.classes_[np.argmax(predictions)]
            
            if max_probability > 0.5:
                predicted_label = f"Person {predicted_label}"
            else:
                predicted_label = "Unknown"

              
            # Save the detected face to a folder named after the person detected
            os.makedirs(predicted_label, exist_ok=True)
            face_img = frame[y:h, x:w]
            cv2.imwrite(f"{predicted_label}/face_{img_counter}.jpg", face_img)
            img_counter += 1
              
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                predicted_label,
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

