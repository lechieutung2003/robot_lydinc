import cv2


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        
    def get_frame(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        skip_frames = int(fps / 3)
        
        for i in range(skip_frames):
            print(f"Frame {i}")
            ret, frame = self.cap.read()
            if not ret:
                print("Error: failed to capture image")
                break
            
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()        
        
cap = Camera()
cap.get_frame()
        
