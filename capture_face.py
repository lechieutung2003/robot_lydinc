import cv2
import os
import numpy as np
from detect import FaceDetector
from datetime import datetime
from utility import PathManager, FaceData


def save_detected_faces_image(image_path, image_save_path, detector):
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    MAX_IMAGE_SIZE = 5000

    for root, dirs, files in os.walk(image_path):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(
                    root, filename
                )  # Sử dụng root để lấy đường dẫn đầy đủ
                try:
                    frame = cv2.imread(full_path)
                    # Tiếp tục xử lý ảnh ở đây nếu không có lỗi
                except cv2.error as e:
                    print(f"Lỗi khi đọc ảnh {filename}: {e}")
                    continue  # Bỏ qua ảnh này và tiếp tục với ảnh tiếp theo

                if frame is None:
                    print(f"Không thể đọc ảnh từ {full_path}. Bỏ qua.")
                    continue  # Chỉ bỏ qua nếu không thể đọc ảnh

                detection_results = detector.detect(
                    frame
                )  # Giả sử detect trả về danh sách các bounding box [(x, y, w, h), ...]

                if detection_results:
                    _, detected_faces = detection_results[
                        -1
                    ]  # Get the last item's detected faces

                    for index, face_data in enumerate(detected_faces):
                        if not face_data:  # Kiểm tra nếu mảng faces trống
                            print(f"No faces detected in result for image {filename}")
                            continue

                        x, y, w, h = face_data.bbox  # Extract bounding box
                        face_image = frame[
                            y:h, x:w
                        ]  # Sửa đoạn này để cắt đúng khuôn mặt

                        if (
                            face_image.size == 0
                        ):  # Kiểm tra nếu face_image rỗng hoặc kích thước không hợp lệ
                            print(
                                f"Error with image: {full_path}"
                            )  # In ra đường dẫn ảnh lỗi
                            continue

                        face_image = cv2.resize(
                            face_image, (256, 256)
                        )  # Resize ảnh về 256x256

                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        save_path = os.path.join(
                            image_save_path, f"face_{timestamp}_{index}.jpg"
                        )
                        cv2.imwrite(save_path, face_image)


def sharpen_image(img, alpha=1.0, beta=1.5, gamma=0.0):
    # Bước 1: Đọc ảnh

    # Bước 2: Resize ảnh (ví dụ: resize về 500x500)
    resized_img = cv2.resize(img, (500, 500))

    # Bước 3: Làm mờ ảnh
    blurred_img = cv2.GaussianBlur(resized_img, (0, 0), 3)

    # Bước 4: Tạo mặt nạ làm nét
    sharpened_img = cv2.addWeighted(resized_img, alpha, blurred_img, beta, gamma)

    return sharpened_img


def save_detected_faces_video(directory_path, image_save_path, detector):
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    for root, dirs, files in os.walk(directory_path):
        # print(f"Processing directory: {root}")
        # Lặp qua mỗi file trong thư mục
        for filename in files:
            print(f"Processing file: {filename}")
            # Kiểm tra phần mở rộng của file để xác định đó có phải là video không
            if filename.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv", "MOV", "MP4", "AVI", "MKV")
            ):
                # Nếu đúng, xử lý video
                video_path = os.path.join(root, filename)
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print("Error opening video file")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps / 3)

                frame_index = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_index % frame_interval == 0:
                        detection_results = detector.detect(frame)

                        if detection_results:
                            _, detected_faces = detection_results[-1]

                            for index, face_data in enumerate(detected_faces):

                                if not face_data:
                                    continue

                                x, y, w, h = face_data.bbox
                                face_image = frame[y:h, x:w]

                                if face_image.size == 0:
                                    continue

                                face_image = cv2.resize(face_image, (256, 256))

                                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                save_path = os.path.join(
                                    image_save_path,
                                    f"face_{timestamp}_{frame_index}_{index}.jpg",
                                )
                                cv2.imwrite(save_path, face_image)

                    frame_index += 1

                cap.release()
                print("Completed processing video.")


if __name__ == "__main__":
    image_path = r"input"
    image_save_path = r"outpu"
    video_path = ""
    image_from_video = ""
    save_detected_faces_image(image_path, image_save_path, FaceDetector())
    # save_detected_faces_video(video_path, image_from_video, FaceDetector())
    print("Hoàn thành việc cắt và lưu khuôn mặt.")
