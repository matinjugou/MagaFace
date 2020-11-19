# MagaFace SDK文档

我们的SDK全部位于`sdk`目录下，主要由个类组成：

- `face_detect_sdk.face_detector.FaceDetector`：检测人脸；
- `face_recognize_sdk.face_recognizer.FaceRecognizer`：抽取人脸特征，并进行比较；
- `face_match_sdk.face_matcher.FaceMatcher`：对上面两个类进行了封装，完成从提取到比对一整个流程。

## 1 `FaceDetector`

示例代码如下：

```python
import cv2

from face_detect_sdk.face_detector import FaceDetector


if __name__ == "__main__":
    image_path = 'data/test.jpg'
    img = cv2.imread(image_path)

    face_detector = FaceDetector()

    # execute detect
    face_detection_list = face_detector.detect_image(img)

    # print
    [print(d) for d in face_detection_list]

    # visualize
    show_image = True
    if show_image:
        img = face_detector.visualize(img, face_detection_list)
        cv2.imshow("", img)
        cv2.waitKey(0)
```

接下来我来介绍各个方法：

- `detect_image(self, img: np.ndarray) -> List[FaceDetection]`：
检测一张图片中的所有人脸，并返回一个`FaceDetection`列表，包含了每个人脸的位置、置信度的数据；
- `detect_images(self, imgs: List[np.ndarray]) -> List[List[FaceDetection]]`：与上面的函数类似，只是会检测一系列的图片；
- `visualize(self, image: np.ndarray, detection_list: List[FaceDetection]) -> None`：在一张图片上标出识别的人脸（一个矩形）。

## 2 `FaceRecognizer`

示例代码如下：

```python
import cv2

from face_recognize_sdk.face_recognizer import FaceRecognizer

if __name__ == "__main__":
    image_path_1 = '../data/1.png'
    img_1 = cv2.imread(image_path_1)
    image_path_2 = '../data/2.png'
    img_2 = cv2.imread(image_path_2)

    face_recognizer = FaceRecognizer()

    # execute verify
    match1 = face_recognizer.verify(img_1, img_2)

    # print
    print(f'face#`{image_path_1}` and face#`{image_path_2}` verification result: {match1}')
```

接下来我来介绍各个方法：

- `generate_feature(self, img: np.ndarray) -> np.ndarray`：针对人脸生成特征；
- `calculate_feature_dist(self, f1: np.ndarray, f2: np.ndarray) -> float:`：对给定的两个特征计算其距离；
- `verify(self, gallery_img: np.ndarray, query_img: np.ndarray, dist_threshold=None) -> bool`：计算两个人脸的相似程度，返回是否是同一个人；
- `verify_feature(self, f1: np.ndarray, f2: np.ndarray, dist_threshold=None) -> (float, bool):`：对给定的两个特征计算其距离，并返回是否是同一个人。


## 3 FaceMatcher

示例代码如下：

```python
import cv2

from face_match_sdk.face_matcher import FaceMatcher

if __name__ == "__main__":
    image_path = 'data/test.jpg'
    img = cv2.imread(image_path)

    face_matcher = FaceMatcher()

    # execute detect
    faces = face_matcher.extract_faces(img)

    # print
    print(f'There are {len(faces)} faces in {image_path}')

    for i in range(len(faces) - 1):
        for j in range(i + 1, len(faces)):
            score, likely = face_matcher.compare_faces(face1=faces[i], face2=faces[j])
            print(f'face {i + 1} and face{j + 1} {"is" if likely else "is not"} the same person. (score = {score})')

    # visualize
    show_image = True
    if show_image:
        img = face_matcher.visualize(img, faces)
        cv2.imshow("", img)
        cv2.waitKey(0)
```

接下来我来介绍各个方法：

- `extract_faces(self, img: np.ndarray) -> List[FeaturedFaceDetection]`：识别图中的人脸，并抽取它们的特征，返回句柄；
- `extract_whole_face(self, img: np.ndarray) -> FeaturedFaceDetection`：将整个图作为人脸，抽取特征，返回句柄；
- `compare_faces(self, face1: FeaturedFaceDetection, face2: FeaturedFaceDetection, threshold=None) -> (float, bool)`：比较两个人脸句柄是否相似，返回相似度，和是否同一人；
- `visualize(self, image: np.ndarray, detection_list: List[FeaturedFaceDetection]) -> None`：在一张图片上标出识别的人脸（一个矩形加一个数字标号）。
