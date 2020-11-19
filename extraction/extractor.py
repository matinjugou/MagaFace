import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox


class Extractor():
    def __init__(self, model_path, device='0', img_size=640, conf_thres=0.25, iou_thres=0.45):
        self.device = select_device(device)
        self.model = attempt_load(model_path, map_location=self.device)
        self.half = self.device.type != 'cpu'
        self.img_size = img_size
        if self.half:
            self.model.half()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def predict_from_file(self, image_path):
        img0 = cv2.imread(image_path)
        return self.predict(img0)
    
    def predict(self, img0):
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        # Inference
        pred = self.model(img)[0]
    
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        ans = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for item in det:
                    ans.append(item.cpu().numpy())
        return ans


if __name__ == "__main__":
    extractor = Extractor(model_path='path/to/best.pt')
    print(extractor.predict_from_file('path/to/image.jpg'))
