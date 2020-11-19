import cv2

from face_recognize_sdk.face_recognizer import FaceRecognizer

if __name__ == "__main__":
    image_path_1 = '../data/1.png'
    img_1 = cv2.imread(image_path_1)
    image_path_2 = '../data/2.png'
    img_2 = cv2.imread(image_path_2)

    face_recognizer = FaceRecognizer()

    # execute verify
    match1 = face_recognizer.verify(img_1, img_2)

    # print
    print(f'face#`{image_path_1}` and face#`{image_path_2}` verification result: {match1}')
