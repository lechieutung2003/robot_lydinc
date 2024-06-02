from pydantic import BaseModel
import base64
import numpy as np
from utility import PathManager, FaceData
import requests
import cv2

class FaceDetector():
    def __init__(self, path_manager = PathManager(), frame = None, detection_threshold=0.5, min_face_size=None, max_face_size=None):
        self.detected_faces = []
        self.path_manager = path_manager
        self.build_request (frame, detection_threshold)
        self.min_face_size = min_face_size
        self.max_face_max = max_face_size
        
    def build_request(self, frame=None, detection_threshold=0.5):
        # Request body for face detection using InsightFace.
        self.req_body = {
            "images": {
                "data": [
                    f"{frame}"
                    ]
        },
        "max size": [
            "720",
            "720"
        ],
        "threshold": f"{detection_threshold}", 
        "embed_only": 'false', 
        "return_face_data": 'false', 
        "return landmarks": 'false', 
        "extract_embedding": 'true', 
        "extract_ga": 'false', 
        "detect_masks": 'false', 
        "limit faces": "0", 
        "min face size": '0', 
        "verbose_timings": 'false', 
        "magpack": 'false',
        "use_rotation": 'false'
        }
        return self.req_body
    
    def detect(self, frame):
        base64_frame = base64.b64encode(cv2.imencode('.jpg', frame) [1]).decode()
        self.build_request(base64_frame)
        detection_req_url = self.path_manager. InsightFace_url + 'extract/'
        response=requests.post(detection_req_url, json=self.req_body).json()
        
        faces= response['data'] [0] ['faces']
        detected_faces= []
        for face in faces:
            #if self.max_face_size > int (face['size']) >= self.min_face_size or (sel
            x,y,w,h = face['bbox']
            embedding = face ['vec']
            face_data = FaceData (id='', embedding=embedding, bbox=[x, y, w, h])
            detected_faces.append(face_data)
        self.detected_faces.append((frame, detected_faces))
        return self.detected_faces