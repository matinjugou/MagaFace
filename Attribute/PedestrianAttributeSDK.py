from PIL import Image

import numpy as np
import torch

from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50

import torchvision.transforms as T
import cv2


height = 274
width = 274
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = T.Compose([
    T.Resize((height, width)),
    T.ToTensor(),
    normalize
])

accessory_type = ["戴帽子", "戴围巾", "没有配饰", "戴墨镜"]
upperBody_style = ["休闲装", "正式装"]
upperBody_dress = ["夹克", "带LOGO的衣服", "格子衫", "短袖衬衫", "薄条纹夹克", "T恤", "其他", "V型领"]
lowerBody_style = ["休闲装", "正式装"]
lowerBody_dress = ["牛仔裤", "短裤", "短裙", "长裤"]
footwear_dress = ["皮鞋", "凉鞋", "普通鞋", "运动鞋"]
carrying_type = ["背包", "其他", "单肩包", "无背包", "塑料袋"]
age_range = ["小于30岁", "大于30岁小于45岁", "大于45岁小于60岁", "大于60岁"]

def get_reload_weight(model_path, model):
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dicts'])
    model = model.module
    return model

class PedestrianAttributeSDK():
    def __init__(self, model_path):
        backbone = resnet50()
        classifier = BaseClassifier(nattr=35)
        model = FeatClassifier(backbone, classifier)
        model = model.cuda()
        self.model = get_reload_weight(model_path, model)
        self.model = self.model.cuda()
        self.model.eval()

    def predict(self, img0):
        '''
            img0 = Image.open('path/to/image.jpg')
        '''
        with torch.no_grad():
            img0 = transform(img0).cuda().unsqueeze(0)
            valid_logits = self.model(img0)
            probs = torch.sigmoid(valid_logits).cpu().numpy()[0]
            return self.translateAttr(probs)
    
    @staticmethod
    def translateAttr(prob):
        return {
            "配饰": accessory_type[np.argmax(prob[0:4])],
            "头发长度": "长发" if prob[4] > 0.5 else "短发",
            "上衣风格": upperBody_style[np.argmax(prob[5:7])],
            "上衣着装": upperBody_dress[np.argmax(prob[7:15])],
            "下身风格": lowerBody_style[np.argmax(prob[15:17])],
            "下身着装": lowerBody_dress[np.argmax(prob[17:21])],
            "鞋子": footwear_dress[np.argmax(prob[21:25])],
            "背包": carrying_type[np.argmax(prob[25:30])],
            "年龄": age_range[np.argmax(prob[30:34])],
            "性别": "男性" if prob[34] > 0.5 else "女性"
        }


if __name__ == "__main__":
    PSDK = PedestrianAttributeSDK("peta_ckpt_max.pth")
    # img = Image.open("test2.jpg").convert('RGB')
    img = cv2.imread("test2.jpg")
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(PSDK.predict(image))
    pass