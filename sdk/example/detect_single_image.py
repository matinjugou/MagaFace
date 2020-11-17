

import cv2

from face_detect_sdk.face_detector import FaceDetector, FaceDetection


if __name__ == "__main__":
    image_path = '../data/test.jpg'
    img = cv2.imread(image_path)

    # TODO: modify this
    face_detector = FaceDetector()

    # execute detect
    face_detection_list = face_detector.detect_image(img)

    # print
    [print(d) for d in face_detection_list]

    # visualize
    show_image = True
    if show_image:
        img = face_detector.visualize(img, face_detection_list)
        cv2.imshow("", img)
        cv2.waitKey(0)
