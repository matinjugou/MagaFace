# Face Recognition SDK


## Usage

```
from recognition import RecognitionModel

model = RecognitionModel(model_path='xxx.pth')

result = model.predict(img_path='xxx.jpg')
```
## Result format

```
result format: [1, 512]
```