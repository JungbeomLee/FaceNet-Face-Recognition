import tensorflow as tf
import mediapipe as mp
import PIL
import numpy as np
import os
import cv2
from typing import Optional

class FaceEmbedder:
    def __init__(self, model_path: str, margin: float = 0.15) :
        self.margin = margin  # Adjust this value to change the cropping margin
        self.model = tf.keras.models.load_model(model_path)
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def _get_face(self, img) -> Optional[np.ndarray]:
        img = img
        face = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not face.detections:
            print('Cant find face')
            return None

        for detection in face.detections:
            # Get bounding box coordinates
            box = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)

            # Calculate margins for x, y, w, h
            mx = int(w * self.margin)
            my = int(h * self.margin)

            # Crop the face with margins
            face_image = img[y+my:y+h-my, x+mx:x+w-mx]
            if face_image.size == 0:
                print('Face image size is null')
                continue

            # Resize the image to 105x105
            return cv2.resize(face_image, (160, 160))

        return None

    def get_embedded_face(self, face_image) -> Optional[np.ndarray]:
        img = cv2.imread(face_image)
        if img is None:
            print(f"Failed to read image: {face_image}")
            return None

        face_image = self._get_face(img)
        if face_image is None:
            return None

        # Preprocess the image
        image = np.expand_dims(face_image, axis=0)  # Add batch dimension
        image = image.astype('float32') / 255  # Normalize to [0,1]

        # Compute embedding
        return self.model.predict(image)[0]
