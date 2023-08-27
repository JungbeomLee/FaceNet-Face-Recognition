import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2

class FaceEmbedder:
    def __init__(self, model_path: str, margin: float = 0.15) :
        self.margin = margin  # Adjust this value to change the cropping margin

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.__input_details = self.interpreter.get_input_details()
        self.__output_details = self.interpreter.get_output_details()

        self.input_index = self.__input_details[0]['index']
        self.output_index = self.__output_details[0]['index']

        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def _get_distance(self, embedded_image1, embedded_image2) :
        distance = np.linalg.norm(embedded_image1 - embedded_image2)
        return distance

    def _get_most_similar_vactor(self, distance_list) :
        # Compute min and max distances
        min_distance = min(distance_list)
        max_distance = max(distance_list)
        similar_images = []

        # Normalize distances and compute similarities
        for i, distance in enumerate(distance_list):
            # Normalize the distance to [0,1] using the min and max distances
            normalized_distance = (distance - 0) / (max_distance - 0)

            # Convert to percentage
            percentage = (1 - normalized_distance) * 100

            # Compute the similarity (as 1 - normalized Euclidean distance)
            similarity = percentage

            # Add new image
            similar_images.append((i, similarity))  # Keeping track of the index and similarity

        # Sort the images by similarity in descending order
        similar_images.sort(key=lambda x: x[1], reverse=True)

        # Return the top 5 most similar images (indices and similarities)
        return similar_images[:5]


    def _get_face(self, img) :
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
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            return cv2.resize(face_image, (160, 160))

        return None

    def get_embedded_face(self, face_image) :
        img = cv2.imread(face_image)
        if img is None:
            print(f"Failed to read image: {face_image}")
            return None

        face_image = self._get_face(img)
        if face_image is None:
            return None

        # Preprocess the image
        image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image.astype('float32') / 255  # Normalize to [0,1]

        # Compute embedding
        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        embeddings = self.interpreter.get_tensor(self.output_index)

        return embeddings