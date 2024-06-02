import cv2
import numpy as np
from joblib import load

# Load the model train
model = load("D:\workspace\digit recognition\RecognitonFace-lbfgs-relu-3-64.pkl")

# Load the model feature extraction
embedder = embedding.InceptionResnetV1(pretrained="vggface2").eval()

# Load the MTCNN to detect face
detector = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")


def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = Image.fromarray(face_img)
    # Transform the image (resize and convert to tensor)
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    face_img = transform(face_img)

    faceEmbed = embedder(face_img.unsqueeze(0))
    flattenEmbed = (
        faceEmbed.squeeze(0).detach().numpy()
    )  # remove batch dimension and convert to numpy array
    return flattenEmbed


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    boxes, _ = detector.detect(frame)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)
            face_img = frame[y:h, x:w]
            preprocessed_face = preprocess_face(face_img)
            # Predict the identity using your pre-trained Scikit-learn model
            predictions = model.predict([preprocessed_face])
            predicted_label = predictions[0]

            # Draw rectangle around the face and put a label
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Person {predicted_label}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Face Recogniton", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
