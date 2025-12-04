# -*- coding: utf-8 -*-
import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer,Qt
import numpy as np
import detect_components

# 如果有 RealSense/ZED/Orbbec SDK 可以在这里导入
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False

try:
    import pyorbbecsdk as ob
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False


class CameraWidget(QWidget):
    def __init__(self, camera_name,on_delete_callback=None, parent=None):
        super().__init__(parent)
        self.camera_name = camera_name
        self.on_delete_callback = on_delete_callback
        self.cap = None
        self.pipeline = None  # 仅 RealSense 用
        self.zed_cam = None   # 仅 ZED 用
        self.orbbec_pipeline = None # 仅 Orbbec 用

        self.label = QLabel("正在初始化...", self)
        self.label.setStyleSheet("background-color: black;")
        self.label.setFixedSize(200, 150)

        self.delete_button = QPushButton("删除", self)
        self.delete_button.setFixedSize(50, 25)
        self.delete_button.clicked.connect(self.on_delete)


        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.top_layout = QHBoxLayout()
        self.top_layout.addWidget(self.label)
        self.top_layout.addWidget(self.delete_button)
        self.top_layout.setAlignment(Qt.AlignTop | Qt.AlignRight)

        layout.addLayout(self.top_layout)

        # 占用摄像头
        if not detect_components.allocate_camera(camera_name, "GUI_Component"):
            self.label.setText("摄像头已被占用")
            return

        # 初始化摄像头
        self.init_camera()

        # 定时更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 约 30fps

    def on_delete(self):
        if self.on_delete_callback:
            self.on_delete_callback() 


    def init_camera(self):
        # ---------------- OpenCV ----------------

        if self.camera_name.startswith("opencv_"):
            idx = int(self.camera_name.split("_")[1])
            print(f"尝试打开 OpenCV 摄像头索引: {idx}")
            
            self.cap = cv2.VideoCapture(idx)
            if not self.cap.isOpened():
                print(f"OpenCV 摄像头 {idx} 打开失败，尝试其他索引...")
                # 尝试其他索引
                for test_idx in range(10):
                    test_cap = cv2.VideoCapture(test_idx)
                    if test_cap.isOpened():
                        print(f"找到可用摄像头索引: {test_idx}")
                        self.cap = test_cap
                        self.camera_name = f"opencv_{test_idx}"
                        test_cap.release()  # 先释放，后面重新打开
                        break
                else:
                    self.label.setText(f"所有OpenCV摄像头都无法打开")
                    self.cap = None
                    return
            
            # 设置摄像头属性
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 验证是否真的能获取帧
            ret, test_frame = self.cap.read()
            if ret:
                print(f"成功获取测试帧，形状: {test_frame.shape}")
            else:
                print("测试帧获取失败")

        
        # ---------------- RealSense ----------------
        elif HAS_REALSENSE and self.camera_name.startswith("RealSense_"):  
            serial = self.camera_name.split("_")[1]  
            ctx = rs.context()  
            device_list = ctx.query_devices()   
            dev = next((d for d in device_list if d.get_info(rs.camera_info.serial_number) == serial), None)  
            if dev:  
                self.pipeline = rs.pipeline()  
                cfg = rs.config()  
                cfg.enable_device(serial)  
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  
                self.pipeline.start(cfg) 
        
        # ---------------- ZED ----------------
        elif HAS_ZED and self.camera_name.startswith("ZED_"):  
            serial = self.camera_name.split("_")[1]  
            init_params = sl.InitParameters()  
            init_params.input.set_from_serial_number(int(serial))  
            self.zed_cam = sl.Camera()  
            status = self.zed_cam.open(init_params)  
            if status != sl.ERROR_CODE.SUCCESS:  
                self.label.setText("ZED打开失败")  
                self.zed_cam = None 
        
        # ---------------- Orbbec ----------------
        elif HAS_ORBBEC and self.camera_name.startswith("Orbbec_"):  
            try:  
                ctx = ob.Context()  
                dev_list = ctx.query_devices()  
                serial = self.camera_name.split("_")[1]  
                  
                # 查找匹配的设备  
                device = next((dev for dev in dev_list   
                              if dev.get_device_info().get_serial_number() == serial), None)  
                  
                if device is None:  
                    self.label.setText("Orbbec设备未找到")  
                    return  
                  
                # 创建 Orbbec Pipeline  
                self.orbbec_pipeline = ob.Pipeline(device)  
                  
                # 创建并配置 Config  
                config = ob.Config()  
                  
                # 获取并启用彩色流  
                profile_list = self.orbbec_pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)  
                if profile_list.get_count() > 0:  
                    color_profile = profile_list.get_default_video_stream_profile()  
                    config.enable_stream(color_profile)  
                else:  
                    self.label.setText("Orbbec无彩色流")  
                    return  
                  
                # 启动 Pipeline  
                self.orbbec_pipeline.start(config)  
                  
            except Exception as e:  
                self.label.setText(f"Orbbec初始化失败: {e}")  
                self.orbbec_pipeline = None 

    def update_frame(self):
        frame = None

        # OpenCV
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            print(f"OpenCV read结果: ret={ret}, frame形状={frame.shape if frame is not None else None}")
            if not ret:
                print("OpenCV读取失败")
                return
        
        # RealSense
        elif self.pipeline:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
        
        # ZED
        elif self.zed_cam:
            image = sl.Mat()
            if self.zed_cam.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed_cam.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
        
        # Orbbec
        elif self.orbbec_pipeline:  
            try:  
                frames = self.orbbec_pipeline.wait_for_frames(100)  # 100ms 超时  
                if frames:  
                    color_frame = frames.get_color_frame()  
                    if color_frame:  
                        frame = np.asanyarray(color_frame.get_data())  
            except Exception as e:  
                print(f"Orbbec获取帧失败: {e}")  
                return 
        
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)

    def closeEvent(self, event):  
        detect_components.release_camera(self.camera_name)  
  
        if self.cap and self.cap.isOpened():  
            self.cap.release()  
        if self.pipeline:  
            self.pipeline.stop()  
        if self.orbbec_pipeline:  
            self.orbbec_pipeline.stop()  
        if self.zed_cam:  
            self.zed_cam.close()  
        event.accept()
