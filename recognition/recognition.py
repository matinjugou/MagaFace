from model import IR_50

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import os

class RecognitionModel():
    def __init__(self, model_path='backbone_ir50_ms1m_epoch63.pth', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        input_size = [112, 112]
        rgb_mean = [0.5, 0.5, 0.5]
        rgb_std = [0.5, 0.5, 0.5]
        embedding_size = 512

        self.transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)])

        self.backbone = IR_50(input_size)
        self.backbone.load_state_dict(torch.load(model_path))
        self.backbone.eval()

        self.device = device

        self.backbone.to(self.device)

    def predict(self, img_path):
        """
            img path
        """
        x = Image.open(img_path).convert('RGB')
        x = self.transform(x).expand(1, 3, 112, 112)
        x = x.to(self.device)
        res = self.backbone(x)
        return res.cpu().detach().numpy()


