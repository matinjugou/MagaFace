# ReIDSDK
用于行人检测的SDK
### 要求
```
torch==1.6.0
```

### 安装
```
pip install -r requirements.txt
cd DCNv2_latest/
./make.sh
```

### 使用方式
```
extractor = ReID('fairmot_dla34.pth')
img0 = cv2.imread('/workspace/huangchao/FairMOT/demos/reid/00046.jpg')
print(extractor.predict(img0))
```

