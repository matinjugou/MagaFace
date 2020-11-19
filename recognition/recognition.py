from backbone.model_irse import IR_50
from util.utils import l2_norm

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import os


class RecognitionModel():
    def __init__(self, model_path='Backbone_IR_50_Epoch_125_Batch_3125_Time_2020-11-19-13-22_checkpoint.pth',
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
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
        self.backbone.load_state_dict(torch.load(model_path, map_location=device))
        self.backbone.eval()

        self.device = device

        # With map_location, following codes seem unnecessary. Preserved to ensure compatibility.
        self.backbone.to(self.device)

    def predict(self, img_path):
        """
            img path
        """
        x = Image.open(img_path).convert('RGB')
        return self.predict_raw(x)

    def predict_raw(self, img):
        x = self.transform(img).expand(1, 3, 112, 112)
        x = x.to(self.device)
        res = self.backbone(x)
        return l2_norm(res.cpu().detach()).numpy()


    def distance(self, f1, f2):
        f1=torch.Tensor(f1)
        f2=torch.Tensor(f2)
        cos = torch.nn.functional.cosine_similarity(f1,f2).detach()
        return cos.numpy()