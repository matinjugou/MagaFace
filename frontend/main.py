import math
import os
import sys
from threading import Condition, Lock

from cv2 import cv2
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QInputDialog

from ui_mainwindow import Ui_MainWindow

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

reid_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Reid'))
dcn_v2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Reid/DCNv2_latest'))
extraction_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../extraction'))
attribute_sdk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Attribute'))
for path in [reid_path, dcn_v2_path, extraction_sdk, attribute_sdk]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ReIDSDK import ReID
from extractor import Extractor
from PedestrianAttributeSDK import PedestrianAttributeSDK

REID_MODEL_PATH = os.path.join(reid_path, 'fairmot_dla34.pth')
EXTRACTOR_MODEL_PATH = os.path.join(extraction_sdk, 'face_extraction.pt')
ATTRIBUTE_MODEL_PATH = os.path.join(attribute_sdk, 'peta_ckpt_max.pth')


class VideoInfoRow:
    File = 0
    Frame = 1  # changeable
    Time = 2  # changeable
    Progress = 3  # changeable
    Width = 4
    Height = 5
    FrameRate = 6
    TotalFrame = 7


class VideoPlayer(QObject):
    frameReady = pyqtSignal(QImage, dict, list)
    finished = pyqtSignal()

    def __init__(self, capture, pkl, reid_model, face_model, attribute_model):
        super(QObject, self).__init__()
        self.capture = capture
        self.pkl = pkl
        self.reid_model = reid_model
        self.capture_mutex = Lock()
        self.running = True  # Python assignment is atomic
        self.playing = False
        self.playing_cv = Condition()
        # Savable data
        # frameIndex -> [{ bbox, re_id, nullable face_bbox, nullable  face_id }]
        self.raw_data = {}

    def thread(self): # A slot takes no params
        with self.playing_cv:
            while self.running:
                self.playing_cv.wait_for(lambda: self.playing)
                if not self.running:
                    break
                with self.capture_mutex:
                    remaining, frame = self.capture.read()
                    data = {
                        VideoInfoRow.Frame: round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)),
                        VideoInfoRow.Time: self.capture.get(cv2.CAP_PROP_POS_MSEC),
                        VideoInfoRow.Progress: self.capture.get(cv2.CAP_PROP_POS_AVI_RATIO),
                    }
                if not remaining:
                    self.finished.emit()
                    break
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                reids = self.reid_model.predict(frame)
                self.frameReady.emit(image, data, reids)
        self.running = False

    def stop(self):
        self.running = False
        if not self.playing:
            self.pause_or_resume(True)

    def pause_or_resume(self, status):
        if self.playing == status:
            return
        self.playing = status
        if status:
            with self.playing_cv:
                self.playing_cv.notify()

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
        if remaining:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frameReady.emit(image, data, [])


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Data
        self.url = None
        self.pkl = None
        self.player_thread = None
        self.player = None
        self.playing = False
        self.total_time_text = '00:00'
        self.slider_dragging = False

        # Models
        self.reid_model = ReID(REID_MODEL_PATH, model_name='dla_34')
        self.face_model = Extractor(EXTRACTOR_MODEL_PATH, 'cpu')
        self.attribute_model = PedestrianAttributeSDK(ATTRIBUTE_MODEL_PATH, 'cpu')

        # Signals and slots
        self.ui.playButton.clicked.connect(lambda: self.play() if self.player is None else self.pause_or_resume())
        self.ui.openFile.triggered.connect(self.open_file)
        self.ui.openStream.triggered.connect(self.open_rtsp)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.showSidebar.triggered.connect(self.toggle_sidebar)
        self.ui.playSlider.sliderPressed.connect(self.slider_pressed)
        self.ui.playSlider.sliderMoved.connect(self.jump_to_frame)
        self.ui.playSlider.sliderReleased.connect(self.slider_released)

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
            self.pkl = os.path.splitext(filename)[0] + '.npy'
            self.play()

    def open_rtsp(self):
        text, ok = QInputDialog.getText(self, 'RTSP URL', 'Enter the RTSP stream url')
        if ok:
            self.url = text
            self.pkl = None
            self.play()

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
        frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        table.item(VideoInfoRow.TotalFrame, 1).setText(str(frames))
        if frames > 0:
            self.ui.playSlider.setMaximum(frames - 1)
            self.ui.playSlider.setEnabled(True)
            total_time = frames / fps
            self.total_time_text = '%02d:%02d' % (total_time // 60, total_time % 60)
        else:
            self.ui.playSlider.setMaximum(0)
            self.ui.playSlider.setEnabled(False)
            self.total_time_text = '∞'

        self.player_thread = QThread()
        self.player = VideoPlayer(capture, self.pkl, self.reid_model)
        self.player.moveToThread(self.player_thread)
        self.player.frameReady.connect(self.on_frame_ready)
        self.player.finished.connect(self.on_player_finished)
        self.player_thread.started.connect(self.player.thread)
        self.player_thread.start()
        self.pause_or_resume()

    def pause_or_resume(self):
        if self.player is not None:
            self.playing = not self.playing
            self.update_playing_state()

    def update_playing_state(self):
        if self.player is not None:
            state = self.playing and not self.slider_dragging
            self.player.pause_or_resume(state)
            self.ui.playButton.setText('⏸' if state else '⏵')

    def on_frame_ready(self, frame: QImage, info, reids):
        video = self.ui.video
        table = self.ui.infoTable
        pixmap = QPixmap.fromImage(frame)
        painter = QPainter()
        painter.begin(pixmap)
        painter.setPen(QPen(QColor(0xff0000), 2))
        for reid in reids:
            bbox = reid['bbox']
            painter.drawRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        painter.end()
        pixmap = pixmap.scaled(video.width(), video.height(), Qt.KeepAspectRatio)
        video.setPixmap(pixmap)
        table.item(VideoInfoRow.Frame, 1).setText(str(info[VideoInfoRow.Frame]))
        table.item(VideoInfoRow.Time, 1).setText('%.2f' % info[VideoInfoRow.Time])
        table.item(VideoInfoRow.Progress, 1).setText('%.2f' % (info[VideoInfoRow.Progress] * 100))
        time = math.floor(info[VideoInfoRow.Time] / 1000)
        if self.ui.playSlider.isEnabled():
            self.ui.playSlider.setValue(info[VideoInfoRow.Frame] - 1)
        self.ui.progressText.setText('%02d:%02d/%s' % (time // 60, time % 60, self.total_time_text))

    def on_player_finished(self):
        self.player_thread.quit()
        self.player_thread.wait()
        self.player = None
        self.player_thread = None
        self.playing = False
        self.ui.playButton.setText('⏵')

    def stop(self):
        if self.player is not None:
            self.player.frameReady.disconnect(self.on_frame_ready)
            self.player.finished.disconnect(self.on_player_finished)
            self.player.stop()
            self.player_thread.quit()
            self.player_thread.wait()
        self.player = None
        self.player_thread = None
        self.playing = False
        self.ui.playButton.setText('⏵')

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
