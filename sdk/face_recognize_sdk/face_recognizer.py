import os
import sys

import torch
from PIL import Image
import numpy as np

recognition_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../recognition'))
if recognition_sdk not in sys.path:
    sys.path.append(recognition_sdk)

from recognition import RecognitionModel


class FaceRecognizer:
    def __init__(self, model_path=None, threshold=0.2,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        if model_path is None:
            model_path = os.path.join(recognition_sdk, 'Backbone_IR_50_Epoch_125_Batch_3125_Time_2020-11-19-13-22_checkpoint.pth')
        self.threshold = threshold
        self.recognizer = RecognitionModel(model_path, device)

    def generate_feature(self, img: np.ndarray) -> np.ndarray:
        return self.recognizer.predict_raw(Image.fromarray(img))

    def calculate_feature_dist(self, f1: np.ndarray, f2: np.ndarray) -> float:
        return self.recognizer.distance(f1,f2)[0]

    def verify(self, gallery_img: np.ndarray, query_img: np.ndarray, dist_threshold=None) -> bool:
        f1 = self.generate_feature(gallery_img)
        f2 = self.generate_feature(query_img)
        return self.verify_feature(f1, f2, dist_threshold)[1]

    def verify_feature(self, f1: np.ndarray, f2: np.ndarray, dist_threshold=None) -> (float, bool):
        dist = self.calculate_feature_dist(f1, f2)
        return dist, dist < (dist_threshold or self.threshold)
