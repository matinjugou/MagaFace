import os
import sys

import torch
from PIL import Image

recognition_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../recognition'))
if recognition_sdk not in sys.path:
    sys.path.append(recognition_sdk)

from recognition import RecognitionModel


class FaceRecognizer:
    def __init__(self, model_path=None, threshold=0.8,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        if model_path is None:
            model_path = os.path.join(recognition_sdk, 'Backbone_IR_50_Epoch_125_Batch_3125_Time_2020-11-19-13-22_checkpoint.pth')
        self.threshold = threshold
        self.recognizer = RecognitionModel(model_path, device)

    def generate_feature(self, img):
        return self.recognizer.predict_raw(Image.fromarray(img))

    def calculate_feature_dist(self, f1, f2) -> float:
        return self.recognizer.distance(f1,f2)

    def verify(self, gallery_img, query_img, dist_threshold=None) -> bool:
        f1 = self.generate_feature(gallery_img)
        f2 = self.generate_feature(query_img)
        dist = self.calculate_feature_dist(f1, f2)
        if dist < (dist_threshold or self.threshold):
            return True
        else:
            return False
