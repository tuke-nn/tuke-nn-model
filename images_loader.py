import os
import cv2


class ImagesLoader:
    @staticmethod
    def load_images(images_path):
        images = []
        for img in os.listdir(images_path):
            img_path = os.path.join(images_path, img)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Image not loaded: {img_path}".encode('utf-8'))
        return images
