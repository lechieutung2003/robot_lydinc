import numpy as np
from deepface import DeepFace
import numpy as np
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import random


def extract_features(img_path, backend, model_name):
    alignment_modes = [True, False]

    face_objs = DeepFace.represent(
        img_path=img_path,
        detector_backend=backend,
        align=alignment_modes[0],
        model_name=model_name,
        enforce_detection=False,
    )

    embedding = face_objs[0].get("embedding", None)
    # Ensure the embedding is returned as a NumPy array
    feature_vector = np.array(embedding) if embedding is not None else None
    return feature_vector


file_path = "D:\\Artificial Intelligence\\git clone\\deepface\\test image\\v1\\IMG_9792.jpg"
backend = "retinaface"
model_name = "Facenet"

feature_vector = extract_features(file_path, backend, model_name)

print(feature_vector)
