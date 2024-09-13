import cv2
from deepface import DeepFace

# Cấu hình các tham số
backend = "retinaface"  # Thay thế bằng backend bạn muốn sử dụng
alignment_modes = [True]  # Thay thế bằng chế độ căn chỉnh bạn muốn sử dụng

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Lưu khung hình tạm thời để xử lý
    img_path = "temp_frame.jpg"
    cv2.imwrite(img_path, frame)

    try:
        # Nhận diện khuôn mặt
        face_objs = DeepFace.represent(
            img_path=img_path,
            detector_backend=backend,
            align=alignment_modes[0],
            enforce_detection=False,
        )

        # Vẽ hình chữ nhật xung quanh khuôn mặt và hiển thị thông tin
        for face in face_objs:
            x, y, w, h = (
                face["region"]["x"],
                face["region"]["y"],
                face["region"]["w"],
                face["region"]["h"],
            )
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                face["dominant_emotion"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

    except Exception as e:
        print(f"Error: {e}")

    # Hiển thị khung hình
    cv2.imshow("Real-time Face Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
