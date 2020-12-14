from models.model import create_model, load_model
from models.decode import mot_decode
from models.utils import _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process

import cv2
import torch
import torch.nn.functional as F
import numpy as np

num_classes = 1
max_per_image = 500
down_ratio = 4

def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    #print('dets',dets[0].keys())
    return dets[0]

def merge_outputs(detections):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


class ReID():
    def __init__(self, model_path, conf_thres=0.4, model_name='dla_34'):
        heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
        head_conv = 256
        self.conf_thres = conf_thres
        self.model = create_model(model_name, heads, head_conv)
        self.model = load_model(self.model, model_path)
        # self.model = self.model.to(torch.device('cuda'))
        self.model.eval()

    def predict(self, img0):
        # img0 = cv2.imread(img_path)  # BGR
        img, _, _, _ = letterbox(img0, height=640, width=640)
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        im_blob = torch.from_numpy(img).unsqueeze(0)

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,'out_height': inp_height // down_ratio,'out_width': inp_width // down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg']
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=500)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = post_process(dets, meta)
        dets = merge_outputs([dets])[1]
        remain_inds = dets[:, 4] > self.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        res = []
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            bbox = [
                int(min(bbox[0], bbox[2])),
                int(min(bbox[1], bbox[3])),
                int(max(bbox[0], bbox[2])),
                int(max(bbox[1], bbox[3])),
            ]
            res.append({
                "reid": id_feature[i],
                "bbox": bbox,
            })
        return res


if __name__ == "__main__":
    extractor = ReID('fairmot_dla34.pth')
    img0 = cv2.imread('/workspace/huangchao/FairMOT/demos/reid/00046.jpg')
    print(extractor.predict(img0))
    pass
