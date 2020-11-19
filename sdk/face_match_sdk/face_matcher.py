from typing import List

import cv2
import numpy as np

from face_detect_sdk.face_detector import FaceDetector, BoundingBox, FaceDetection
from face_recognize_sdk.face_recognizer import FaceRecognizer


class FeaturedFaceDetection:
    bbox: BoundingBox
    category: int
    confidence: float
    image: np.ndarray
    feature: np.ndarray

    def __init__(self, detection: FaceDetection, image: np.ndarray, feature: np.ndarray):
        self.bbox = detection.bbox
        self.category = detection.category
        self.confidence = detection.confidence
        self.image = image
        self.feature = feature


class FaceMatcher:
    def __init__(self, detector_model_path=None, recognizer_model_path=None):
        self.detector = FaceDetector(detector_model_path)
        self.recognizer = FaceRecognizer(recognizer_model_path)

    def extract_faces(self, img: np.ndarray) -> List[FeaturedFaceDetection]:
        """
        Detect, crop, and calculate the feature of all faces in the img
        :param img: image in numpy array
        :return: a list of FeaturedFaceDetection objects in image
        """
        faces = self.detector.detect_image(img)
        results = []
        for face in faces:
            image = img[face.bbox.top:face.bbox.bottom, face.bbox.left:face.bbox.right, :]
            feature = self.recognizer.generate_feature(image)
            results.append(FeaturedFaceDetection(face, image, feature))
        return results

    def extract_whole_face(self, img: np.ndarray) -> FeaturedFaceDetection:
        feature = self.recognizer.generate_feature(img)
        return FeaturedFaceDetection(FaceDetection(0, 0, img.shape[2], img.shape[1], 0, 1.0), img, feature)

    def compare_faces(self, face1: FeaturedFaceDetection, face2: FeaturedFaceDetection, threshold=None) -> (float, bool):
        return self.recognizer.verify_feature(face1.feature, face2.feature, dist_threshold=threshold)

    def visualize(self, image: np.ndarray, detection_list: List[FeaturedFaceDetection], color=(0,0,255), thickness=2) -> None:
        img = image.copy()
        for i, detection in enumerate(detection_list):
            bbox = detection.bbox
            p1 = bbox.left, bbox.top
            p2 = bbox.right, bbox.bottom
            cv2.rectangle(img, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.putText(img, text= str(i+1), org=((bbox.left + bbox.right) // 2 - 10, bbox.top - 10),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color,
                        thickness=thickness, lineType=cv2.LINE_AA)
        return img
