import matplotlib.pyplot as plt
import cv2

embeding = FaceEmbedder(model_path = '/content/drive/MyDrive/A.아이/github/models_folder/facenet/facenet_model.h5')
friends_path_first = '/content/drive/MyDrive/A.아이/images/friends_image/fist_image/'
friends_path_first_name = os.listdir(friends_path_first)

friends_path_second = '/content/drive/MyDrive/A.아이/images/friends_image/second_image/'
friends_path_second_name = os.listdir(friends_path_second)

def test_model_with_friends(index, image_friends_first_list, image_friends_second_list, face_embedder):
    # Get the embedding for the anchor image
    anchor_image = friends_path_first+image_friends_first_list[index]
    anchor_embedding = face_embedder.get_embedded_face(anchor_image)

    # Variables to keep track of most similar images
    distances = []  # A list to keep track of the distances

    # Iterate over images to compute distances
    for image in image_friends_second_list:
        # Get the embedding for this image
        image_embedding = face_embedder.get_embedded_face(friends_path_second+image)

        # Compute the distance
        if image_embedding is not None:
            distance = face_embedder._get_distance(anchor_embedding, image_embedding)
            distances.append(distance)
        else:
            distances.append(float('inf'))  # Handle the case where the face is not detected
    # Get the most similar images
    most_similar_images = face_embedder._get_most_similar_vactor(distances)

    # Show the anchor image
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(anchor_image), cv2.COLOR_BGR2RGB))
    plt.title("Anchor image")
    plt.axis('off')

    # Show the most similar images
    for i, (img_index, similarity) in enumerate(most_similar_images):
        similar_image = cv2.imread(friends_path_second+image_friends_second_list[img_index])
        plt.subplot(1, 6, i + 2)
        plt.imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Similarity: {similarity:.5f}%')
        plt.axis('off')
    plt.show()


test_model_with_friends(6, friends_path_first_name, friends_path_second_name, embeding)