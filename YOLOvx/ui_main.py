# -*- coding: utf-8 -*-
"""
基于 YOLOv11 的课堂行为检测系统 - PyQt5 图形界面
集成新训练的 7 类课堂行为模型（50轮）
"""
import sys
import os
import cv2
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox,
    QGroupBox, QFrame, QScrollArea, QSizePolicy, QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from ultralytics import YOLO

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "class_50ep", "weights", "best.pt")

# 模型内置英文类别名
CLASSES_EN = [
    "sleep", "raise_head", "using_computer", "stand",
    "using_phone", "raise_hand", "read",
]
# 中文对照 + 显示颜色（BGR->QColor）
CLASS_INFO = {
    "sleep":          ("睡觉",   "#4A90E2"),
    "raise_head":     ("抬头",   "#27AE60"),
    "using_computer": ("用电脑", "#8E44AD"),
    "stand":          ("站立",   "#E67E22"),
    "using_phone":    ("玩手机", "#E74C3C"),
    "raise_hand":     ("举手",   "#F39C12"),
    "read":           ("读书",   "#1ABC9C"),
}

STYLE = """
QMainWindow, QWidget { background: #F4F6F9; color: #2C3E50; }
QGroupBox {
    font-weight: bold; border: 1px solid #D8DEE9; border-radius: 8px;
    margin-top: 10px; padding: 10px; background: #FFFFFF;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #2C3E50; }
QPushButton {
    background: #FFFFFF; border: 1px solid #C9D2DD; border-radius: 6px;
    padding: 9px; font-weight: bold; color: #2C3E50;
}
QPushButton:hover { background: #EAF2FB; border-color: #4A90E2; }
QPushButton:pressed { background: #D6E7FA; }
QPushButton#danger { background: #FDEDEC; border-color: #E74C3C; color: #C0392B; }
QPushButton#primary { background: #4A90E2; border-color: #4A90E2; color: white; }
QPushButton#primary:hover { background: #3A7BC8; }
QLabel#video { background: #1E272E; border-radius: 8px; }
QSlider::groove:horizontal { height: 6px; background: #D8DEE9; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -5px 0; border-radius: 8px;
    background: #4A90E2; border: 2px solid white;
}
QScrollArea { background: transparent; border: none; }
QTextEdit { background: #FFFFFF; border: 1px solid #D8DEE9; border-radius: 6px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于 YOLOv11 的课堂行为检测系统")
        self.resize(1280, 800)
        self.setStyleSheet(STYLE)

        self.model = None
        self.conf = 0.25
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.mode = None           # 'camera' / 'video'
        self.frame_count = 0
        self.last_saved = ""

        self._build_ui()

        # 加载模型
        self.statusBar().showMessage("正在加载模型...")
        QApplication.processEvents()
        try:
            self.model = YOLO(MODEL_PATH)
            self.model.fuse()
            self.statusBar().showMessage(
                "模型加载成功 | %s | %d 类" % (os.path.basename(MODEL_PATH), len(self.model.names))
            )
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", str(e))
            self.statusBar().showMessage("模型加载失败: %s" % e)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # ============ 左侧控制面板 ============
        left = QFrame()
        left.setFixedWidth(290)
        left.setStyleSheet("QFrame { background: #FFFFFF; border-radius: 10px; }")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 14, 14, 14)
        lv.setSpacing(12)

        title = QLabel("课堂行为检测系统")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        lv.addWidget(title)
        sub = QLabel("YOLOv11 · 7 类行为识别")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#7F8C8D; font-size:11px;")
        lv.addWidget(sub)
        lv.addWidget(self._hline())

        # --- 检测模式 ---
        grp_mode = QGroupBox("检测模式")
        gm = QVBoxLayout(grp_mode)
        self.btn_image = QPushButton("图片检测")
        self.btn_camera = QPushButton("摄像头实时检测")
        self.btn_video = QPushButton("视频文件检测")
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.setObjectName("danger")
        self.btn_image.clicked.connect(self.open_image)
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.start_video)
        self.btn_stop.clicked.connect(self.stop_all)
        for b in (self.btn_image, self.btn_camera, self.btn_video, self.btn_stop):
            gm.addWidget(b)
        lv.addWidget(grp_mode)

        # --- 置信度 ---
        grp_conf = QGroupBox("置信度阈值")
        gc = QVBoxLayout(grp_conf)
        self.conf_label = QLabel("0.25")
        self.conf_label.setAlignment(Qt.AlignRight)
        self.conf_label.setStyleSheet("color:#4A90E2; font-weight:bold;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 90)
        self.slider.setValue(25)
        self.slider.valueChanged.connect(self.on_slider)
        gc.addWidget(self.conf_label)
        gc.addWidget(self.slider)
        lv.addWidget(grp_conf)

        # --- 类别图例 ---
        grp_cls = QGroupBox("行为类别（7类）")
        gcl = QVBoxLayout(grp_cls)
        for en in CLASSES_EN:
            cn, color = CLASS_INFO[en]
            row = QLabel("  ●  %s  (%s)" % (en, cn))
            row.setStyleSheet("color:%s; font-weight:bold;" % color)
            gcl.addWidget(row)
        lv.addWidget(grp_cls)

        self.btn_save = QPushButton("保存当前画面")
        self.btn_save.clicked.connect(self.save_frame)
        lv.addWidget(self.btn_save)
        lv.addStretch()
        root.addWidget(left)

        # ============ 右侧显示区 ============
        right = QVBoxLayout()
        right.setSpacing(12)

        self.video_label = QLabel("请选择检测模式", self)
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFont(QFont("Microsoft YaHei", 14))
        self.video_label.setStyleSheet(
            "QLabel#video { background:#1E272E; color:#7F8C8D; border-radius:10px; }"
        )
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.video_label, stretch=3)

        # --- 当前帧统计 + 日志 ---
        bottom = QFrame()
        bottom.setStyleSheet("QFrame { background:#FFFFFF; border-radius:10px; }")
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(12, 10, 12, 10)

        stat_title = QLabel("本帧检测结果")
        stat_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        bl.addWidget(stat_title)
        self.stat_label = QLabel("等待检测...")
        self.stat_label.setWordWrap(True)
        self.stat_label.setStyleSheet("color:#2C3E50; font-size:12px;")
        bl.addWidget(self.stat_label)
        bl.addWidget(self._hline())

        log_title = QLabel("检测日志")
        log_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        bl.addWidget(log_title)
        self.log_label = QLabel("")
        self.log_label.setWordWrap(True)
        self.log_label.setAlignment(Qt.AlignTop)
        self.log_label.setStyleSheet("color:#566573; font-size:11px;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.log_label)
        bl.addWidget(scroll)
        right.addWidget(bottom, stretch=1)

        root.addLayout(right, stretch=1)
        self.setStatusBar(QStatusBar())

    def _hline(self):
        h = QFrame()
        h.setFrameShape(QFrame.HLine)
        h.setStyleSheet("color:#E5E9EF;")
        h.setFixedHeight(1)
        return h

    # ============ 交互 ============
    @pyqtSlot(int)
    def on_slider(self, v):
        self.conf = v / 100.0
        self.conf_label.setText("%.2f" % self.conf)

    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        cur = self.log_label.text()
        self.log_label.setText("[%s] %s\n%s" % (ts, text, cur)[:2000])

    def _check_model(self):
        if self.model is None:
            QMessageBox.warning(self, "提示", "模型未加载，无法检测")
            return False
        return True

    def open_image(self):
        if not self._check_model():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", BASE_DIR, "图片 (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        try:
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "错误", "无法读取图片")
                return
            results = self.model(img, conf=self.conf, imgsz=640, verbose=False)
            self.last_saved = ""
            self._show_result(img, results[0], os.path.basename(path))
            self.log("图片检测: %s" % os.path.basename(path))
        except Exception as e:
            QMessageBox.critical(self, "检测失败", str(e))

    def start_camera(self):
        if not self._check_model():
            return
        self.stop_all()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "摄像头不可用", "未检测到摄像头或已被占用")
            self.cap = None
            return
        self.mode = "camera"
        self.timer.start(30)   # ~33fps
        self.btn_camera.setText("摄像头检测中...")
        self.log("摄像头实时检测已启动")

    def start_video(self):
        if not self._check_model():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", BASE_DIR, "视频 (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self.stop_all()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件")
            self.cap = None
            return
        self.mode = "video"
        self.timer.start(30)
        self.btn_video.setText("视频检测中...")
        self.log("视频检测: %s" % os.path.basename(path))

    def stop_all(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = None
        self.btn_camera.setText("摄像头实时检测")
        self.btn_video.setText("视频文件检测")

    def _tick(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_all()
            self.log("视频播放结束")
            return
        try:
            results = self.model(frame, conf=self.conf, imgsz=640, verbose=False)
            self.frame_count += 1
            # 摄像头模式每 60 帧记一次日志，避免刷屏
            if self.frame_count % 60 == 0:
                self.log("实时检测中... 已处理 %d 帧" % self.frame_count)
            self._show_result(frame, results[0], "实时画面", quiet=True)
        except Exception as e:
            self.log("检测异常: %s" % e)

    def _show_result(self, frame, result, name, quiet=False):
        """在界面上显示检测结果 + 更新统计"""
        plotted = result.plot()                       # BGR with boxes
        rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        target = self.video_label.size()
        self.video_label.setPixmap(
            pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # 统计
        counts = {}
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                en = result.names.get(cls_id, str(cls_id))
                counts[en] = counts.get(en, 0) + 1
        if counts:
            parts = []
            for en, n in counts.items():
                cn, color = CLASS_INFO.get(en, (en, "#2C3E50"))
                parts.append('<span style="color:%s"><b>%s(%s)</b></span>: %d' % (color, en, cn, n))
            self.stat_label.setText("检出 %d 个目标 &nbsp;|&nbsp; " % sum(counts.values()) + " &nbsp; ".join(parts))
            if not quiet:
                self.log("检出: " + ", ".join("%s x%d" % (k, v) for k, v in counts.items()))
        else:
            self.stat_label.setText("未检出目标（可调低置信度阈值）")
        self.current_frame = plotted

    def save_frame(self):
        if not hasattr(self, "current_frame"):
            QMessageBox.information(self, "提示", "还没有检测结果可保存")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存画面", os.path.join(BASE_DIR, "detect_result.jpg"), "图片 (*.jpg *.png)")
        if path:
            cv2.imwrite(path, self.current_frame)
            self.log("已保存: %s" % os.path.basename(path))

    def closeEvent(self, event):
        self.stop_all()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
