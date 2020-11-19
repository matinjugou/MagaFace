### Face Extraction SDK
Face extraction SDK based on YOLOv5.

### 训练、测试、预测
适用于vscode的`launch.json`文件已经放在了目录下，请自行配置执行环境，需要Pytorch1.6及以上执行环境

#### 使用方式
直接预测图片的结果
```python
import Extractor from extractor

extractor_obj = Extractor(model_path='path/to/weights.pt')
print(extractor_obj.predict_from_file('path/to/image.jpg'))
```

预测RGB格式的numpy对象
```python
import Extractor from extractor

image_array = np.array()
extractor_obj = Extractor(model_path='path/to/weights.pt')
print(extractor_obj.predict(image_array))
```

返回格式
```python
[{x1, y1, x2, y2, confidence, class}]  # 左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标，置信度，类别
'''
example:
[array([        914,         102,        1056,         279,     0.88281,           0], dtype=float32), array([        540,         231,         664,         452,     0.84131,           0], dtype=float32)]
'''
```
