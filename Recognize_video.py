import cv2
import numpy as np
from detect import FaceDetector
from joblib import load

# Load the models
modelNN = load(".\\model\\databaseV4\\model_adam_relu_3_64.joblib")
modelSVM = load(".\\model\\databaseV4\\modelSVM.joblib")

# Load InsightFace model
detector = FaceDetector()

# Open the video file or capture device
video_path = (
    "path_to_your_video.mp4"  # Replace with your video file path or use 0 for webcam
)
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the video writer
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
