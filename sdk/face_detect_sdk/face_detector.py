from typing import List
import cv2
import os
import sys

import torch

extraction_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../extraction'))
if extraction_sdk not in sys.path:
    sys.path.append(extraction_sdk)

from extractor import Extractor


class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int

    def __init__(self, left, top, right, bottom):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)


class FaceDetection:
    bbox: BoundingBox
    category: int
    confidence: float

    def __init__(self, left, top, right, bottom, category, confidence):
        self.bbox = BoundingBox(left, top, right, bottom)
        self.category = int(category)
        self.confidence = confidence

    def __str__(self):
        return f'#FaceDetection# bbox=[({self.bbox.left},{self.bbox.top})->({self.bbox.right},{self.bbox.bottom})], confidence={self.confidence}'


class FaceDetector:
    def __init__(self, model_path=None, device="0" if torch.cuda.is_available() else "cpu"):
        if model_path is None:
            model_path = os.path.join(extraction_sdk, 'face_extraction.pt')
        self.extractor = Extractor(model_path, device)

    def detect_image(self, img) -> List[FaceDetection]:
        return [
            FaceDetection(int(result[0]), int(result[1]),
                          int(result[2]), int(result[3]),
                          int(result[5]), result[4])
            for result in self.extractor.predict(img)
        ]

    def detect_images(self, imgs) -> List[List[FaceDetection]]:
        return [self.detect_image(img) for img in imgs]

    def visualize(self, image, detection_list: List[FaceDetection], color=(0,0,255), thickness=5):
        img = image.copy()
        for detection in detection_list:
            bbox = detection.bbox
            p1 = bbox.left, bbox.top
            p2 = bbox.right, bbox.bottom
            cv2.rectangle(img, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
        return img
