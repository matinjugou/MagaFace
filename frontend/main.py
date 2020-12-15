import pickle
import random
import time
from collections import OrderedDict
from uuid import uuid4

import math
import os
import sys
from threading import Condition, Lock

import numpy as np
from PIL import Image
from cv2 import cv2
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QStaticText, QBrush, QGuiApplication
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QInputDialog, QTableWidgetItem

from ui_mainwindow import Ui_MainWindow

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

reid_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Reid'))
dcn_v2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Reid/DCNv2_latest'))
detect_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../extraction'))
recognize_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../recognition'))
attribute_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Attribute'))
for path in [reid_path, dcn_v2_path, detect_sdk, recognize_sdk, attribute_sdk]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ReIDSDK import ReID
from extractor import Extractor
from recognition import RecognitionModel
from PedestrianAttributeSDK import PedestrianAttributeSDK

REID_MODEL_PATH = os.path.join(reid_path, 'fairmot_dla34.pth')
DETECT_MODEL_PATH = os.path.join(detect_sdk, 'face_extraction.pt')
RECOGNIZE_MODEL_PATH = os.path.join(recognize_sdk, 'Backbone_IR_50_Epoch_125_Batch_3125_Time_2020-11-19-13-22_checkpoint.pth')
ATTRIBUTE_MODEL_PATH = os.path.join(attribute_sdk, 'peta_ckpt_max.pth')

INSTANCE_FEATURE_LENGTH = 128
FACE_FEATURE_LENGTH = 512

INSTANCE_SIMILARITY_THRESHOLD = 0.25
FACE_SIMILARITY_THRESHOLD = 0.25
FEATURE_DECAY = 0.8

FONT_SIZE = 10
FONT_SPACE = 4


class VideoInfoRow:
    File = 0
    Frame = 1  # changeable
    Time = 2  # changeable
    Progress = 3  # changeable
    Width = 4
    Height = 5
    FrameRate = 6
    TotalFrame = 7


# cosine similarity for matrix
def cosine(u: np.array, v: np.array) -> np.array:  # u: m x f, v: n x f
    product = np.dot(u, np.transpose(v))  # m x n
    norm = np.dot(np.linalg.norm(u, axis=1)[:, np.newaxis], np.linalg.norm(v, axis=1)[np.newaxis, :])
    return 1.0 - product / norm


def random_color() -> int:
    return random.randint(0, 0xffffff)


class VideoPlayer(QObject):
    # image, image info, [{ frame: index, instance: uuid, instance_color: rgb,
    #                       instance_name: option[str], instance_bbox: bbox, face: option[uuid],
    #                       face_color: option[rgb], face_bbox: option[bbox], face_name: option[str],
    #                       attributes: dict, blacklist: option[uuid], blacklist_name: option[uuid] }]
    frameReady = pyqtSignal(QImage, dict, list)
    resultsReset = pyqtSignal(list)
    blacklistsReset = pyqtSignal(OrderedDict)
    finished = pyqtSignal(dict)

    def __init__(self, capture, instance_file, reid_model: ReID, detect_model: Extractor,
                 recognize_model: RecognitionModel, attribute_model: PedestrianAttributeSDK):
        super(QObject, self).__init__()
        self.capture = capture
        self.instance_file = instance_file
        self.blacklist_file = None
        self.reid_model = reid_model
        self.detect_model = detect_model
        self.recognize_model = recognize_model
        self.attribute_model = attribute_model
        self.capture_mutex = Lock()
        self.running = True  # Python assignment is atomic
        self.playing = False
        self.playing_cv = Condition()
        self.last_time = None
        # State
        self.detector_enabled = False
        # Results
        # index -> [{ instance: uuid, instance_feature: ndarray, instance_bbox: bbox,
        #             face: option[uuid], face_feature: option[ndarray], face_bbox: option[bbox], attributes: dict }]
        self.results_mutex = Lock()
        self.frames = {}
        # uuid -> { color: rgb, name: option[str] }
        self.instances = OrderedDict()
        self.instance_features = np.empty((0, INSTANCE_FEATURE_LENGTH))
        # uuid -> { color: rgb, name: option[str] }
        self.faces = OrderedDict()
        self.face_features = np.empty((0, FACE_FEATURE_LENGTH))
        # Blacklists
        # uuid -> { name: option[str] }
        self.blacklist = OrderedDict()
        self.blacklist_features = np.empty((0, FACE_FEATURE_LENGTH))

    def load_instance_file(self):
        if not self.instance_file:
            return
        with open(self.instance_file, 'rb') as f:
            data = pickle.load(f)
        with self.results_mutex:
            self.frames = data['frames']
            self.instances = data['instances']
            self.instance_features = data['instance_features']
            self.faces = data['faces']
            self.face_features = data['face_features']
            results = [self.generate_results(frame_index) for frame_index in self.frames]
            self.resultsReset.emit(results)

    def change_instance_name(self, uuid, name):
        with self.results_mutex:
            self.instances[uuid]['name'] = name

    def change_face_name(self, uuid, name):
        with self.results_mutex:
            self.faces[uuid]['name'] = name

    def change_blacklist_name(self, uuid, name):
        with self.results_mutex:
            self.blacklist[uuid]['name'] = name

    def save_instance_file(self):
        if not self.instance_file:
            return
        with self.results_mutex:
            with open(self.instance_file, 'wb') as f:
                pickle.dump({
                    'frames': self.frames,
                    'instances': self.instances,
                    'instance_features': self.instance_features,
                    'faces': self.faces,
                    'face_features': self.face_features,
                }, f)

    def load_blacklist_file(self, blacklist_filename):
        pass

    def save_blacklist_file(self):
        pass

    def set_detector_enabled(self, value):
        self.detector_enabled = value

    def thread(self):
        self.last_time = None
        with self.playing_cv:
            while self.running:
                self.playing_cv.wait_for(lambda: self.playing)
                if not self.running:
                    break
                with self.capture_mutex:
                    rate = 1 / self.capture.get(cv2.CAP_PROP_FPS)
                    new_time = time.time()
                    if self.last_time is not None and new_time < self.last_time + rate:
                        time.sleep(self.last_time + rate - new_time)  # it is hard to sleep with a condition variable
                    self.last_time = new_time
                    frame_index = round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
                    remaining, frame = self.capture.read()
                    data = {
                        VideoInfoRow.Frame: frame_index,
                        VideoInfoRow.Time: self.capture.get(cv2.CAP_PROP_POS_MSEC),
                        VideoInfoRow.Progress: self.capture.get(cv2.CAP_PROP_POS_AVI_RATIO),
                    }
                    width = round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if not remaining:
                    self.finished.emit(data)
                    break
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                if not self.detector_enabled:
                    self.frameReady.emit(image, data, [])
                else:
                    with self.results_mutex:
                        if frame_index not in self.frames:
                            def valid_bbox(bbox):
                                return bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] < width and bbox[3] < height
                            # Get all instances
                            instances = [{
                                'instance_feature': instance['reid'],
                                'instance_bbox': instance['bbox'],
                            } for instance in self.reid_model.predict(frame) if valid_bbox(instance['bbox'])]
                            if instances:
                                features = np.stack([instance['instance_feature'] for instance in instances])
                                similarity = cosine(features, self.instance_features)
                                while similarity.size:
                                    matched = np.unravel_index(np.argmin(similarity), similarity.shape)
                                    if similarity[matched] >= INSTANCE_SIMILARITY_THRESHOLD:
                                        break
                                    uuid = list(self.instances)[matched[1]]
                                    instances[matched[0]]['instance'] = uuid
                                    self.instance_features[matched[1]] = \
                                        FEATURE_DECAY * self.instance_features[matched[1]] + \
                                        (1 - FEATURE_DECAY) * features[matched[0]]
                                    # self.instance_features[matched[1]] = np.average(np.stack(
                                    #     [instance['instance_feature'] for frame in self.frames.values()
                                    #      for instance in frame if instance['instance'] == uuid] +
                                    #     [features[matched[0]]]),
                                    #     axis=0)
                                    similarity[matched[0], :] = np.full(similarity.shape[1], 2)  # 2 is largest similarity
                                    similarity[:, matched[1]] = np.full(similarity.shape[0], 2)
                            for instance in instances:
                                if 'instance' not in instance:
                                    new_uuid = uuid4()
                                    new_instance = {
                                        'color': random_color(),
                                        'name': None,
                                    }
                                    self.instances[new_uuid] = new_instance
                                    instance['instance'] = new_uuid
                                    self.instance_features = np.concatenate(
                                        (self.instance_features, instance['instance_feature'][np.newaxis, :]))
                            faces_instance_index = []
                            for i, instance in enumerate(instances):
                                bbox = instance['instance_bbox']
                                instance_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                instance_image_rgb = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
                                attributes = self.attribute_model.predict(Image.fromarray(instance_image_rgb).convert('RGB'))
                                instance['attributes'] = attributes
                                faces = self.detect_model.predict(instance_image)
                                if faces:
                                    face_bbox = max(faces, key=lambda x: x[4])
                                    face_bbox = [round(face_bbox[0]), round(face_bbox[1]),
                                                 round(face_bbox[2]), round(face_bbox[3])]
                                    face_image = instance_image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
                                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                                    feature = self.recognize_model.predict_raw(Image.fromarray(face_image)).squeeze(0)
                                    faces_instance_index.append(i)
                                    instance['face_feature'] = feature
                                    instance['face_bbox'] = [
                                        bbox[0] + face_bbox[0],
                                        bbox[1] + face_bbox[1],
                                        bbox[0] + face_bbox[2],
                                        bbox[1] + face_bbox[3],
                                    ]
                                else:
                                    instance['face'] = None
                                    instance['face_feature'] = None
                                    instance['face_bbox'] = None
                            if faces_instance_index:
                                features = np.stack([instances[faces_instance_index]['face_feature']
                                                     for faces_instance_index in faces_instance_index])
                                similarity = cosine(features, self.face_features)
                                while similarity.size:
                                    matched = np.unravel_index(np.argmin(similarity), similarity.shape)
                                    if similarity[matched] >= FACE_SIMILARITY_THRESHOLD:
                                        break
                                    uuid = list(self.faces)[matched[1]]
                                    instances[faces_instance_index[matched[0]]]['face'] = uuid
                                    self.face_features[matched[1]] = FEATURE_DECAY * self.face_features[matched[1]] + \
                                                                     (1 - FEATURE_DECAY) * features[matched[0]]
                                    # self.face_features[matched[1]] = np.average(np.stack(
                                    #     [instance['face_feature'] for frame in self.frames.values()
                                    #      for instance in frame if instance['face'] == uuid] +
                                    #     [features[matched[0]]]),
                                    #     axis=0)
                                    similarity[matched[0], :] = np.full(similarity.shape[1], 2)  # 2 is largest similarity
                                    similarity[:, matched[1]] = np.full(similarity.shape[0], 2)
                            for instance in instances:
                                if 'face' in instance or 'face_feature' not in instance:
                                    continue
                                new_uuid = uuid4()
                                new_face = {
                                    'color': random_color(),
                                    'name': None,
                                }
                                self.faces[new_uuid] = new_face
                                instance['face'] = new_uuid
                                self.face_features = np.concatenate(
                                    (self.face_features, instance['face_feature'][np.newaxis, :]))
                            self.frames[frame_index] = instances
                        results = self.generate_results(frame_index)
                    self.frameReady.emit(image, data, results)
        self.running = False

    def generate_results(self, frame_index):
        results = []
        if frame_index in self.frames:
            for item in self.frames[frame_index]:
                instance = self.instances[item['instance']]
                face = None if item['face'] is None else self.faces[item['face']]
                blacklist = None
                if item['face_feature'] is not None and self.blacklist_features.shape[0] > 0:
                    face_feature = item['face_feature']
                    similarity = cosine(face_feature[np.newaxis, :], self.blacklist_features).squeeze(0)
                    index = np.argmin(similarity)
                    if similarity[index] < FACE_SIMILARITY_THRESHOLD:
                        blacklist = list(self.blacklist)[index]
                results.append({
                    'frame': frame_index,
                    'instance': item['instance'],
                    'instance_feature': item['instance'],
                    'instance_color': instance['color'],
                    'instance_name': instance['name'],
                    'instance_bbox': item['instance_bbox'],
                    'face': item['face'],
                    'face_color': None if face is None else face['color'],
                    'face_name': None if face is None else face['name'],
                    'face_bbox': item['face_bbox'],
                    'attributes': item['attributes'].copy(),
                    'blacklist': blacklist,
                    'blacklist_name': None if blacklist is None else self.blacklist[blacklist]['name'],
                })
        return results

    def stop(self):
        self.running = False
        if not self.playing:
            self.pause_or_resume(True)

    def pause_or_resume(self, status):
        if self.playing == status:
            return
        self.playing = status
        with self.playing_cv:
            self.playing_cv.notify_all()

    def set_frame(self, pos):
        with self.capture_mutex:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
            remaining, frame = self.capture.read()
            data = {
                VideoInfoRow.Frame: round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)),
                VideoInfoRow.Time: self.capture.get(cv2.CAP_PROP_POS_MSEC),
                VideoInfoRow.Progress: self.capture.get(cv2.CAP_PROP_POS_AVI_RATIO),
            }
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self.last_time = None
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        with self.results_mutex:
            results = self.generate_results(pos)
        self.frameReady.emit(image, data, results)


def pretty_frames(frames):
    results = []
    head = None
    previous = None
    i = None
    for i in sorted(frames):
        if head is None:
            head = i
        elif previous + 1 < i:
            results.append((head, previous))
            head = i
        else:
            assert previous + 1 == i
        previous = i
    if head is not None and i is not None:
        results.append((head, i))
    return ', '.join([str(i) if head == i else '%s-%s' % (head, i) for head, i in results])


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Data
        self.url = None
        self.instance_file = None
        self.player_thread = None
        self.player = None
        self.playing = False
        self.finished = True
        self.total_time_text = '00:00'
        self.total_frames = 0
        self.slider_dragging = False
        self.frames = set()
        self.items = []
        self.instances = OrderedDict()
        self.faces = OrderedDict()

        # Models
        self.reid_model = ReID(REID_MODEL_PATH, model_name='dla_34')
        self.face_detect_model = Extractor(DETECT_MODEL_PATH, 'cpu')
        self.face_recognize_model = RecognitionModel(RECOGNIZE_MODEL_PATH)
        self.attribute_model = PedestrianAttributeSDK(ATTRIBUTE_MODEL_PATH, 'cpu')

        # Signals and slots
        self.ui.playButton.clicked.connect(lambda: self.play() if self.finished else self.pause_or_resume())
        self.ui.openFile.triggered.connect(self.open_file)
        self.ui.openStream.triggered.connect(self.open_rtsp)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.showSidebar.triggered.connect(self.toggle_sidebar)
        self.ui.playSlider.sliderPressed.connect(self.slider_pressed)
        self.ui.playSlider.sliderMoved.connect(self.jump_to_frame)
        self.ui.playSlider.sliderReleased.connect(self.slider_released)
        self.ui.playSlider.valueChanged.connect(self.jump_to_frame)
        self.ui.detector.changed.connect(self.set_detector_enabled)
        self.ui.saveResults.clicked.connect(self.save_results)
        self.ui.actionSaveResults.triggered.connect(self.save_results)
        self.ui.pedestrianTable.cellChanged.connect(self.on_pedestrian_cell_changed)
        self.ui.pedestrianTable.cellClicked.connect(self.on_pedestrian_cell_clicked)
        self.ui.instanceTable.cellChanged.connect(self.on_instance_cell_changed)
        self.ui.faceTable.cellChanged.connect(self.on_face_cell_changed)

    def on_pedestrian_cell_changed(self, row, column):
        table_item = self.ui.pedestrianTable.item(row, column)
        item = self.items[row]
        if column == 1:  # instance ID
            text = table_item.text() or None
            if self.player:
                self.player.change_instance_name(item[0], text)
            self.ui.instanceTable.item(self.instances[item[0]][0], 1).setText(text or '')
            for i in self.instances[item[0]][1]:
                self.ui.pedestrianTable.item(i, column).setText(text or str(item[0])[:8])
        elif column == 3:  # face ID
            if item[1] is None:
                return
            text = table_item.text() or None
            if self.player:
                self.player.change_face_name(item[1], text)
            self.ui.faceTable.item(self.faces[item[1]][0], 1).setText(text or '')
            for i in self.faces[item[1]][1]:
                self.ui.pedestrianTable.item(i, column).setText(text or str(item[1])[:8])
        elif column == 5:  # blacklist
            pass

    def on_pedestrian_cell_clicked(self, row, column):
        if not QGuiApplication.keyboardModifiers() == Qt.ControlModifier:
            return
        item = self.items[row]
        if column == 1:  # instance ID
            self.ui.pedestrianTabWidget.setCurrentIndex(1)
            table_item = self.ui.instanceTable.item(self.instances[item[0]][0], 0)
            self.ui.instanceTable.scrollToItem(table_item)
            self.ui.instanceTable.setCurrentItem(table_item)
        elif column == 3:  # face ID
            if item[1] is None:
                return
            self.ui.pedestrianTabWidget.setCurrentIndex(2)
            table_item = self.ui.faceTable.item(self.faces[item[1]][0], 0)
            self.ui.faceTable.scrollToItem(table_item)
            self.ui.faceTable.setCurrentItem(table_item)

    def on_instance_cell_changed(self, row, column):
        table_item = self.ui.instanceTable.item(row, column)
        uuid, (_, rows, _) = list(self.instances.items())[row]
        if column == 1:
            text = table_item.text() or None
            if self.player:
                self.player.change_instance_name(uuid, text)
            self.ui.instanceTable.item(row, 1).setText(text or '')
            for i in rows:
                self.ui.pedestrianTable.item(i, 1).setText(text or str(uuid)[:8])

    def on_face_cell_changed(self, row, column):
        table_item = self.ui.faceTable.item(row, column)
        uuid, (_, rows, _) = list(self.faces.items())[row]
        if column == 1:
            text = table_item.text() or None
            if self.player:
                self.player.change_face_name(uuid, text)
            self.ui.faceTable.item(row, 1).setText(text or '')
            for i in rows:
                self.ui.pedestrianTable.item(i, 3).setText(text or str(uuid)[:8])

    def slider_pressed(self):
        self.slider_dragging = True
        self.update_playing_state()

    def slider_released(self):
        self.slider_dragging = False
        self.update_playing_state()

    def jump_to_frame(self, value):
        if self.player is not None:
            self.player.set_frame(value)

    def toggle_sidebar(self, value):
        self.ui.splitter.setSizes([int(value), 1])

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select Video File',
                                                  '.', 'Video Files (*.mp4 *.flv *.ts *.mts *.avi)')
        if filename != '':
            self.url = filename
            self.instance_file = os.path.splitext(filename)[0] + '.pkl'
            self.play()

    def open_rtsp(self):
        text, ok = QInputDialog.getText(self, 'RTSP URL', 'Enter the RTSP stream url')
        if ok:
            self.url = text
            self.instance_file = None
            self.play()

    def set_detector_enabled(self):
        if self.player:
            self.player.set_detector_enabled(self.ui.detector.isChecked())

    def play(self):
        if self.url is None:
            return
        capture = cv2.VideoCapture(self.url)
        if not capture.isOpened():
            QMessageBox.critical(self, "Open File Error", "Failed to open file")
            return

        self.stop()

        table = self.ui.infoTable
        table.item(VideoInfoRow.File, 1).setText(self.url)
        table.item(VideoInfoRow.Width, 1).setText(str(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        table.item(VideoInfoRow.Height, 1).setText(str(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fps = capture.get(cv2.CAP_PROP_FPS)
        table.item(VideoInfoRow.FrameRate, 1).setText(str(round(fps, 2)))
        self.total_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        table.item(VideoInfoRow.TotalFrame, 1).setText(str(self.total_frames))
        if self.total_frames > 0:
            self.ui.playSlider.setMaximum(self.total_frames - 1)
            self.ui.playSlider.setEnabled(True)
            total_time = self.total_frames / fps
            self.total_time_text = '%02d:%02d' % (total_time // 60, total_time % 60)
        else:
            self.total_frames = 0
            self.ui.playSlider.setMaximum(0)
            self.ui.playSlider.setEnabled(False)
            self.total_time_text = '∞'
        self.frames = set()
        self.items = []
        self.instances = OrderedDict()
        self.faces = OrderedDict()
        self.ui.pedestrianTable.setRowCount(0)
        self.ui.instanceTable.setRowCount(0)
        self.ui.faceTable.setRowCount(0)

        self.player_thread = QThread()
        self.player = VideoPlayer(capture, self.instance_file, self.reid_model, self.face_detect_model,
                                  self.face_recognize_model, self.attribute_model)
        self.player.moveToThread(self.player_thread)
        self.player.frameReady.connect(self.on_frame_ready)
        self.player.finished.connect(self.on_player_finished)
        self.player.resultsReset.connect(self.reset_results)
        self.player_thread.started.connect(self.player.thread)
        self.player_thread.start()
        self.player.set_detector_enabled(self.ui.detector.isChecked())
        if self.instance_file and os.path.exists(self.instance_file):
            self.player.load_instance_file()
        self.finished = False
        self.pause_or_resume()

    def reset_results(self, results):
        self.frames = set()
        self.items = []
        self.instances = OrderedDict()
        self.faces = OrderedDict()
        self.ui.pedestrianTable.setRowCount(0)
        self.ui.instanceTable.setRowCount(0)
        self.ui.faceTable.setRowCount(0)
        for result in results:
            self.process_results(result)

    def process_results(self, results):
        if not results:
            return
        frame_index = results[0]['frame'] + 1
        if frame_index in self.frames:
            return
        self.frames.add(frame_index)
        last_pedestrian_item = None
        last_instance_item = None
        last_face_item = None
        self.ui.pedestrianTable.blockSignals(True)
        self.ui.instanceTable.blockSignals(True)
        self.ui.faceTable.blockSignals(True)
        for result in results:
            self.items.append((result['instance'], result['face'], result['blacklist']))
            # Instance
            if result['instance'] not in self.instances:
                row = self.ui.instanceTable.rowCount()
                self.ui.instanceTable.insertRow(row)
                self.instances[result['instance']]= row, set(), set()
                last_instance_item = QTableWidgetItem(str(result['instance']))
                last_instance_item.setFlags(last_instance_item.flags() & ~Qt.ItemIsEditable)
                self.ui.instanceTable.setItem(row, 0, last_instance_item)
                item = QTableWidgetItem(result['instance_name'] or '')
                # item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.ui.instanceTable.setItem(row, 1, item)
                item = QTableWidgetItem()
                item.setBackground(QColor(result['instance_color']))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.ui.instanceTable.setItem(row, 2, item)
                item = QTableWidgetItem()
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.ui.instanceTable.setItem(row, 3, item)
            # Face
            if result['face'] is not None and result['face'] not in self.faces:
                row = self.ui.faceTable.rowCount()
                self.ui.faceTable.insertRow(row)
                self.faces[result['face']]= row, set(), set()
                last_face_item = QTableWidgetItem(str(result['face']))
                last_face_item.setFlags(last_face_item.flags() & ~Qt.ItemIsEditable)
                self.ui.faceTable.setItem(row, 0, last_face_item)
                item = QTableWidgetItem(result['face_name'] or '')
                # item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.ui.faceTable.setItem(row, 1, item)
                item = QTableWidgetItem()
                item.setBackground(QColor(result['face_color']))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.ui.faceTable.setItem(row, 2, item)
                item = QTableWidgetItem()
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.ui.faceTable.setItem(row, 3, item)
            # Pedestrian
            row = self.ui.pedestrianTable.rowCount()
            self.ui.pedestrianTable.insertRow(row)
            last_pedestrian_item = QTableWidgetItem(str(frame_index))
            last_pedestrian_item.setFlags(last_pedestrian_item.flags() & ~Qt.ItemIsEditable)
            self.ui.pedestrianTable.setItem(row, 0, last_pedestrian_item)
            item = QTableWidgetItem(result['instance_name'] or str(result['instance'])[:8])
            # item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.pedestrianTable.setItem(row, 1, item)
            item = QTableWidgetItem()
            item.setBackground(QColor(result['instance_color']))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.pedestrianTable.setItem(row, 2, item)
            item = QTableWidgetItem('' if result['face'] is None else result['face_name'] or str(result['face'])[:8])
            if result['face'] is None:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.pedestrianTable.setItem(row, 3, item)
            item = QTableWidgetItem()
            if result['face_color'] is not None:
                item.setBackground(QColor(result['face_color']))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.pedestrianTable.setItem(row, 4, item)
            item = QTableWidgetItem('' if result['blacklist'] is None else
                                    result['blacklist_name'] or str(result['blacklist'])[:8])
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.pedestrianTable.setItem(row, 5, item)
            item = QTableWidgetItem(', '.join(['%s: %s' % (key, value) for key, value in result['attributes'].items()]))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.pedestrianTable.setItem(row, 6, item)
            # Update instance frames
            instance_row, pedestrians, frames = self.instances[result['instance']]
            if frame_index not in frames:
                pedestrians.add(row)
                frames.add(frame_index)
                self.ui.instanceTable.item(instance_row, 3).setText(pretty_frames(frames))
                self.instances[result['instance']][1].add(row)
            # Update face frames
            if result['face'] is not None:
                face_row, pedestrians, frames = self.faces[result['face']]
                if frame_index not in frames:
                    pedestrians.add(row)
                    frames.add(frame_index)
                    self.ui.faceTable.item(face_row, 3).setText(pretty_frames(frames))
                    self.faces[result['face']][1].add(row)
        if last_pedestrian_item is not None:
            self.ui.pedestrianTable.scrollToItem(last_pedestrian_item)
        if last_instance_item is not None:
            self.ui.instanceTable.scrollToItem(last_instance_item)
        if last_face_item is not None:
            self.ui.faceTable.scrollToItem(last_face_item)
        self.ui.pedestrianTable.blockSignals(False)
        self.ui.instanceTable.blockSignals(False)
        self.ui.faceTable.blockSignals(False)

    def pause_or_resume(self):
        if not self.finished:
            self.playing = not self.playing
            self.update_playing_state()

    def update_playing_state(self):
        if self.player is not None:
            state = self.playing and not self.slider_dragging
            self.player.pause_or_resume(state)
            self.ui.playButton.setText('⏸' if state else '⏵')

    def on_frame_ready(self, frame: QImage, info, instances):
        self.process_results(instances)
        video = self.ui.video
        table = self.ui.infoTable
        origin_pixmap = QPixmap.fromImage(frame)
        painter = QPainter()
        pixmap = origin_pixmap.scaled(video.width(), video.height(), Qt.KeepAspectRatio)
        painter.begin(pixmap)
        width_scale = pixmap.width() / origin_pixmap.width()
        height_scale = pixmap.height() / origin_pixmap.height()
        def translate_bbox(bbox):
            return [width_scale * bbox[0], height_scale * bbox[1], width_scale * bbox[2], height_scale * bbox[3]]
        font = painter.font()
        font.setPointSize(FONT_SIZE)
        painter.setFont(font)
        for instance in instances:
            color = instance['instance_color']
            painter.setPen(QPen(QColor(color), 4))
            bbox = translate_bbox(instance['instance_bbox'])
            painter.drawRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            if instance['face'] is not None:
                painter.setPen(QPen(QColor(instance['face_color']), 2))
                bbox = translate_bbox(instance['face_bbox'])
                painter.drawRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        for instance in instances:
            bbox = translate_bbox(instance['instance_bbox'])
            attributes = {
                **instance['attributes'],
                'ID': instance['instance_name'] or str(instance['instance'])[:8],
            }
            height = len(attributes) * (FONT_SIZE + FONT_SPACE) + 2 * FONT_SPACE
            top = bbox[1] - height
            background_color = QColor(0xffffff)
            background_color.setAlphaF(0.5)
            painter.fillRect(bbox[0], top, 150, height, background_color)
            painter.setPen(QPen(QColor(0x0)))
            top += FONT_SPACE
            for key, value in attributes.items():
                painter.drawStaticText(bbox[0] + FONT_SPACE, top, QStaticText('%s: %s' % (key, value)))
                top += FONT_SIZE + FONT_SPACE
        for instance in instances:
            if instance['face'] is None:
                continue
            bbox = translate_bbox(instance['face_bbox'])
            attributes = {
                'ID': instance['face_name'] or str(instance['face'])[:8],
            }
            if instance['blacklist'] is not None:
                attributes['黑名单'] = instance['blacklist_name'] or str(instance['blacklist'])[:8]
            height = len(attributes) * (FONT_SIZE + FONT_SPACE) + 2 * FONT_SPACE
            top = bbox[3]
            background_color = QColor(0xffffff)
            background_color.setAlphaF(0.5)
            painter.fillRect(bbox[0], top, 100, height, background_color)
            painter.setPen(QPen(QColor(0x0)))
            top += FONT_SPACE
            for key, value in attributes.items():
                painter.drawStaticText(bbox[0] + FONT_SPACE, top, QStaticText('%s: %s' % (key, value)))
                top += FONT_SIZE + FONT_SPACE
        painter.end()
        video.setPixmap(pixmap)
        table.item(VideoInfoRow.Frame, 1).setText(str(info[VideoInfoRow.Frame]))
        table.item(VideoInfoRow.Time, 1).setText('%.2f' % info[VideoInfoRow.Time])
        table.item(VideoInfoRow.Progress, 1).setText('%.2f' % (info[VideoInfoRow.Progress] * 100))
        time = math.floor(info[VideoInfoRow.Time] / 1000)
        if self.ui.playSlider.isEnabled():
            self.ui.playSlider.blockSignals(True)
            self.ui.playSlider.setValue(info[VideoInfoRow.Frame] - 1)
            self.ui.playSlider.blockSignals(False)
        self.ui.progressText.setText('%02d:%02d/%s %d/%d' % (time // 60, time % 60, self.total_time_text,
                                                             info[VideoInfoRow.Frame], self.total_frames))

    def save_results(self):
        if self.player is not None:
            self.player.save_instance_file()
            self.ui.statusBar.showMessage('Results saved!')

    def on_player_finished(self, info):
        self.player.save_instance_file()
        self.playing = False
        self.finished = True
        table = self.ui.infoTable
        table.item(VideoInfoRow.Frame, 1).setText(str(info[VideoInfoRow.Frame]))
        table.item(VideoInfoRow.Time, 1).setText('%.2f' % info[VideoInfoRow.Time])
        table.item(VideoInfoRow.Progress, 1).setText('%.2f' % (info[VideoInfoRow.Progress] * 100))
        time = math.floor(info[VideoInfoRow.Time] / 1000)
        self.ui.playButton.setText('⏵')
        self.ui.playSlider.setEnabled(False)
        self.ui.playSlider.setValue(info[VideoInfoRow.Frame] - 1)
        self.ui.progressText.setText(
            '%02d:%02d/%s %d/%d' % (time // 60, time % 60, self.total_time_text, self.total_frames, self.total_frames))

    def stop(self):
        if self.player is not None:
            self.player.frameReady.disconnect(self.on_frame_ready)
            self.player.finished.disconnect(self.on_player_finished)
            self.player.stop()
        if self.player_thread is not None:
            self.player_thread.quit()
            self.player_thread.wait()
        self.player = None
        self.player_thread = None
        self.playing = False
        self.finished = True
        self.ui.playButton.setText('⏵')

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
