#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度相机多视图演示程序
同时显示RGB、IR、Depth图像，并集成MediaPipe人体骨架检测
基于Scepter SDK和现有的深度处理代码
"""
import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

# MediaPipe 导入和初始化
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_MODE = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_MODE = "TASKS_API"
    print(f"MediaPipe {mp.__version__} Tasks API 导入成功")
except ImportError as e:
    print(f"MediaPipe Tasks API 不可用: {e}")
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_MODE = "LEGACY_API"
        print(f"MediaPipe {mp.__version__} Legacy API 导入成功")
    except ImportError as e2:
        print(f"MediaPipe 不可用: {e2}")
        MEDIAPIPE_AVAILABLE = False
        MEDIAPIPE_MODE = None

# 导入现有的工具和SDK
try:
    # 添加当前目录和related_code目录到路径，以便导入其他模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ad_depth_dir = os.path.join(current_dir, "related_code")

    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if ad_depth_dir not in sys.path:
        sys.path.insert(0, ad_depth_dir)

    # 设置SDK路径到sys.path
    sdk_python_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "MultilanguageSDK", "Python")
    if os.path.exists(sdk_python_path) and sdk_python_path not in sys.path:
        sys.path.insert(0, sdk_python_path)

    # 导入SDK加载模块
    from sdk_loader import load_scepter_sdk
    # 导入相机管理模块
    from camera_manager import initialize_camera as _initialize_camera, open_and_start_stream as _open_and_start_stream
    # 导入工具函数
    from utils import convert_depthframe_to_array, normalize_depth_to_8bit
    # 导入帧处理模块
    from frame_processor import process_frame_from_array, visualize_pointcloud

    SDK_AVAILABLE = True
    print("SDK模块导入成功")
except ImportError as e:
    print(f"导入模块失败: {e}")
    print(f"当前路径: {current_dir}")
    print(f"related_code路径: {ad_depth_dir}")
    print(f"SDK Python路径: {sdk_python_path}")
    print(f"Python路径: {sys.path[:3]}")  # 只显示前3个路径
    SDK_AVAILABLE = False

class MultiViewDepthCameraDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-View Depth Camera Demo with Skeleton")
        self.root.geometry("1600x900")

        # 检查SDK路径
        self.check_sdk_path()

        # 相机相关变量
        self.camera = None
        self.device_info = None
        self.depth_max = None
        self.camera_param = None
        self.camera_actual_fps = None
        self.actual_rgb_resolution = None
        self.ScFrameType = None

        # MediaPipe相关变量
        self.pose_detector = None
        self.pose_connections = None
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        self.mediapipe_mode = MEDIAPIPE_MODE

        if self.mediapipe_available:
            try:
                if self.mediapipe_mode == "TASKS_API":
                    # 使用新版本的MediaPipe Tasks API
                    model_path = self._get_pose_model()
                    base_options = python.BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=python.BaseOptions.Delegate.CPU
                    )

                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,  # 使用IMAGE模式，每帧独立处理
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        output_segmentation_masks=False,
                        num_poses=1
                    )

                    self.pose_detector = vision.PoseLandmarker.create_from_options(options)
                    self.pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
                    print("MediaPipe Pose detector (Tasks API) initialized successfully")

                elif self.mediapipe_mode == "LEGACY_API":
                    # 使用旧版本的MediaPipe API
                    import mediapipe as mp
                    self.mp_pose = mp.solutions.pose
                    self.mp_drawing = mp.solutions.drawing_utils
                    self.mp_drawing_styles = mp.solutions.drawing_styles

                    self.pose_detector = self.mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    print("MediaPipe Pose detector (Legacy API) initialized successfully")

            except Exception as e:
                print(f"MediaPipe初始化失败: {e}")
                self.pose_detector = None
                self.mediapipe_available = False

        # 线程控制
        self.running = False
        self.camera_thread = None
        self.update_thread = None

        # 图像显示变量
        self.rgb_image_label = None
        self.ir_image_label = None
        self.depth_image_label = None

        # 状态变量
        self.connected = False

        # 保存相关
        import datetime

        # 创建GUI界面
        self.create_gui()

    def _get_pose_model(self):
        """获取或下载MediaPipe Pose模型"""
        try:
            import urllib.request
            from pathlib import Path

            model_dir = Path(__file__).parent / 'models'
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / 'pose_landmarker.task'

            if model_path.exists():
                print(f"Using existing pose model: {model_path}")
                return model_path

            # 下载MediaPipe Pose模型
            model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

            print("Downloading MediaPipe pose model...")
            print("This may take a few minutes...")

            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model downloaded to: {model_path}")

            return model_path

        except Exception as e:
            print(f"Error getting pose model: {e}")
            raise

    def check_sdk_path(self):
        """检查并设置SDK路径"""
        try:
            # 设置SDK路径环境变量 - 指向MultilanguageSDK/Python目录
            sdk_python_path = os.path.join(os.path.dirname(__file__), "related_code", "scepter_sdk", "ScepterSDK-master", "MultilanguageSDK", "Python")
            sdk_base_path = os.path.join(os.path.dirname(__file__), "related_code", "scepter_sdk", "ScepterSDK-master", "BaseSDK")

            if os.path.exists(sdk_python_path):
                os.environ['SCEPTER_SDK_PATH'] = sdk_python_path
                print(f"设置SDK Python路径: {sdk_python_path}")
            else:
                print(f"SDK Python路径不存在: {sdk_python_path}")

            if os.path.exists(sdk_base_path):
                # 设置库路径
                lib_path = os.path.join(sdk_base_path, "AArch64", "Lib")
                if os.path.exists(lib_path):
                    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                    if current_ld_path:
                        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}"
                    else:
                        os.environ['LD_LIBRARY_PATH'] = lib_path
                    print(f"设置库路径: {lib_path}")
                else:
                    print(f"库路径不存在: {lib_path}")
            else:
                print(f"SDK Base路径不存在: {sdk_base_path}")

        except Exception as e:
            print(f"设置SDK路径失败: {e}")

        # Check if SDK is available
        if not SDK_AVAILABLE:
            messagebox.showerror("Error", "SDK modules not available, please ensure all dependencies are installed")
            return

    def create_gui(self):
        """创建GUI界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))

        # Connect button
        self.connect_btn = ttk.Button(control_frame, text="Connect Camera", command=self.connect_camera)
        self.connect_btn.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        # Disconnect button
        self.disconnect_btn = ttk.Button(control_frame, text="Disconnect", command=self.disconnect_camera, state=tk.DISABLED)
        self.disconnect_btn.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))

        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))

        self.save_btn = ttk.Button(button_frame, text="Save Screenshot", command=self.save_screenshot, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=0, padx=(0, 5))

        self.clear_btn = ttk.Button(button_frame, text="Clear Display", command=self.clear_display)
        self.clear_btn.grid(row=0, column=1)

        # MediaPipe状态显示
        mediapipe_frame = ttk.LabelFrame(control_frame, text="MediaPipe Status", padding="5")
        mediapipe_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))

        if self.mediapipe_available:
            if self.mediapipe_mode == "TASKS_API":
                mediapipe_status = "Available (Tasks API)"
            elif self.mediapipe_mode == "LEGACY_API":
                mediapipe_status = "Available (Legacy API)"
            else:
                mediapipe_status = "Available (Unknown)"
        else:
            mediapipe_status = "Not Available"

        self.mediapipe_label = ttk.Label(mediapipe_frame, text=f"MediaPipe: {mediapipe_status}")
        self.mediapipe_label.grid(row=0, column=0, sticky=tk.W)

        # Status Display
        status_frame = ttk.LabelFrame(control_frame, text="Camera Status", padding="5")
        status_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(status_frame, text="Not Connected")
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        # Display Area - Four-view layout (2x2 grid)
        display_frame = ttk.LabelFrame(main_frame, text="Four-View Display", padding="5")
        display_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置显示区域的网格权重 (2x2)
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)

        # RGB Display (top-left)
        rgb_frame = ttk.LabelFrame(display_frame, text="RGB", padding="2")
        rgb_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5), pady=(0, 5))
        rgb_frame.columnconfigure(0, weight=1)
        rgb_frame.rowconfigure(0, weight=1)

        self.rgb_image_label = ttk.Label(rgb_frame)
        self.rgb_image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # IR Display (top-right)
        ir_frame = ttk.LabelFrame(display_frame, text="IR", padding="2")
        ir_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        ir_frame.columnconfigure(0, weight=1)
        ir_frame.rowconfigure(0, weight=1)

        self.ir_image_label = ttk.Label(ir_frame)
        self.ir_image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Depth Display (bottom-left)
        depth_frame = ttk.LabelFrame(display_frame, text="Depth", padding="2")
        depth_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        depth_frame.columnconfigure(0, weight=1)
        depth_frame.rowconfigure(0, weight=1)

        self.depth_image_label = ttk.Label(depth_frame)
        self.depth_image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Skeleton Alert Display (bottom-right)
        skeleton_frame = ttk.LabelFrame(display_frame, text="Status Alert", padding="2")
        skeleton_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        skeleton_frame.columnconfigure(0, weight=1)
        skeleton_frame.rowconfigure(0, weight=1)

        self.status_canvas = tk.Canvas(skeleton_frame, highlightthickness=0)
        self.status_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text = self.status_canvas.create_text(0, 0, text="NO PERSON", fill="white", font=("Arial", 24, "bold"))

        # Internal state for alert flashing
        self.person_detected = False
        self.alert_flash_state = False


    def check_camera_available(self):
        """检查相机是否可用"""
        try:
            # 加载SDK
            sdk_loaded, sdk_base_path, ScepterTofCam, ScSensorType, ScFrameType, ScWorkMode, ScExposureControlMode, ScTimeFilterParams, ScConfidenceFilterParams, ScFlyingPixelFilterParams = load_scepter_sdk()
            if not sdk_loaded:
                return False, "SDK加载失败"

            # 创建临时相机实例检查设备数量
            temp_camera = ScepterTofCam()
            camera_count = temp_camera.scGetDeviceCount(3000)
            if camera_count <= 0:
                return False, "No camera device found, please ensure camera is connected and powered on"

            return True, f"找到 {camera_count} 个相机设备"

        except Exception as e:
            return False, f"相机检查失败: {str(e)}"

    def connect_camera(self):
        """连接相机"""
        try:
            self.status_label.config(text="Checking camera...")
            self.connect_btn.config(state=tk.DISABLED)

            # 首先检查相机是否可用
            available, msg = self.check_camera_available()
            if not available:
                raise RuntimeError(msg)

            self.status_label.config(text="Connecting to camera...")

            # 加载SDK
            sdk_loaded, sdk_base_path, ScepterTofCam, ScSensorType, ScFrameType, ScWorkMode, ScExposureControlMode, ScTimeFilterParams, ScConfidenceFilterParams, ScFlyingPixelFilterParams = load_scepter_sdk()
            if not sdk_loaded:
                raise RuntimeError("SDK加载失败")

            self.ScFrameType = ScFrameType

            # 初始化相机
            self.camera, self.device_info, device_already_open = _initialize_camera(ScepterTofCam)
            if not self.camera or not self.device_info:
                raise RuntimeError("相机初始化失败")

            # 启动流
            self.depth_max, self.camera_param, self.camera_actual_fps, self.actual_rgb_resolution = _open_and_start_stream(
                self.camera, self.device_info, ScSensorType, ScWorkMode,
                enable_rgb=True, rgb_resolution=(640, 480), enable_ir=True, target_fps=30,
                ScTimeFilterParams=ScTimeFilterParams, ScConfidenceFilterParams=ScConfidenceFilterParams,
                ScFlyingPixelFilterParams=ScFlyingPixelFilterParams, ScExposureControlMode=ScExposureControlMode,
                device_already_open=device_already_open
            )

            self.connected = True
            self.status_label.config(text=f"Connected - FPS: {self.camera_actual_fps}")
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)

            # 启动相机数据获取线程
            self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            # 启动UI更新线程
            self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
            self.update_thread.start()

        except Exception as e:
            messagebox.showerror("Connection Failed", f"Camera connection failed: {str(e)}")
            self.status_label.config(text="Connection Failed")
            self.connect_btn.config(state=tk.NORMAL)

    def disconnect_camera(self):
        """断开相机连接"""
        try:
            self.running = False
            self.status_label.config(text="Disconnecting...")

            # 等待线程结束
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2.0)

            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2.0)

            # 关闭相机
            if self.camera:
                try:
                    self.camera.scStopStream()
                    self.camera.scCloseDevice()
                except Exception as e:
                    print(f"Error closing camera: {e}")

            self.camera = None
            self.connected = False
            self.status_label.config(text="Disconnected")
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)

            # 清空显示
            self.clear_display()

        except Exception as e:
            messagebox.showerror("Disconnect Failed", f"Camera disconnect failed: {str(e)}")

    def camera_loop(self):
        """相机数据获取循环"""
        while self.running and self.connected:
            try:
                # 检查帧是否准备好
                ret, frameready = self.camera.scGetFrameReady(1200)
                if ret != 0:
                    continue

                # 获取深度帧
                if frameready.depth:
                    ret, depth_frame = self.camera.scGetFrame(self.ScFrameType.SC_DEPTH_FRAME)
                    if ret == 0:
                        self.process_depth_frame(depth_frame)
                    else:
                        print(f"Get depth frame failed: {ret}")

                # 获取IR帧
                if frameready.ir:
                    ret, ir_frame = self.camera.scGetFrame(self.ScFrameType.SC_IR_FRAME)
                    if ret == 0:
                        self.process_ir_frame(ir_frame)
                    else:
                        print(f"Get IR frame failed: {ret}")

                # 获取RGB帧
                if frameready.color:
                    ret, rgb_frame = self.camera.scGetFrame(self.ScFrameType.SC_COLOR_FRAME)
                    if ret == 0:
                        self.process_rgb_frame(rgb_frame)
                    else:
                        print(f"Get RGB frame failed: {ret}")

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"Camera loop error: {e}")
                self.status_label.config(text=f"Camera error: {str(e)}")
                break

    def process_depth_frame(self, frame):
        """Process depth frame"""
        try:
            # Save frame for point cloud conversion
            self.depth_frame = frame

            # Convert to numpy array
            depth_array = convert_depthframe_to_array(frame)

            # Save depth data for display
            self.depth_data = depth_array
            self.depth_frame_info = {
                'width': frame.width,
                'height': frame.height
            }

        except Exception as e:
            print(f"Process depth frame error: {e}")

    def process_ir_frame(self, frame):
        """Process IR frame"""
        try:
            # Convert IR to numpy array
            ir_array = np.ctypeslib.as_array(frame.pFrameData, (1, frame.dataLen))
            ir_array.dtype = np.uint8
            ir_array = ir_array.reshape((frame.height, frame.width))

            # Save IR data for display
            self.ir_data = ir_array

        except Exception as e:
            print(f"Process IR frame error: {e}")

    def process_rgb_frame(self, frame):
        """处理RGB帧"""
        try:
            # 转换RGB数据为numpy数组
            rgb_array = np.ctypeslib.as_array(frame.pFrameData, (1, frame.width * frame.height * 3))
            rgb_array.dtype = np.uint8
            rgb_array = rgb_array.reshape((frame.height, frame.width, 3))

            # SDK直接输出BGR格式，无需转换
            self.rgb_data = rgb_array

        except Exception as e:
            print(f"Process RGB frame error: {e}")

    def process_pose_detection(self, detect_image, draw_image):
        """
        Perform pose detection using MediaPipe.
        Args:
            detect_image: Image used for detection (RGB is best)
            draw_image: Image used for drawing the skeleton (Depth visualization)
        """
        try:
            has_person = False

            if self.mediapipe_mode == "TASKS_API":
                has_person = self._process_pose_tasks_api(detect_image, draw_image)
            elif self.mediapipe_mode == "LEGACY_API":
                has_person = self._process_pose_legacy_api(detect_image, draw_image)
            else:
                raise ValueError("Unknown MediaPipe mode")

            self.person_detected = has_person

        except Exception as e:
            print(f"Pose detection error: {e}")
            self.person_detected = False

    def _process_pose_tasks_api(self, detect_image, draw_image):
        """Pose detection using MediaPipe Tasks API"""
        # MediaPipe requires RGB
        rgb_detect = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_detect)

        # Detect
        detection_result = self.pose_detector.detect(mp_image)
        has_person = False

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            has_person = True
            landmarks = detection_result.pose_landmarks[0]
            # Draw skeleton on the target draw_image (Depth map)
            self._draw_pose_landmarks_tasks(draw_image, landmarks)

        return has_person

    def _process_pose_legacy_api(self, detect_image, draw_image):
        """Pose detection using MediaPipe Legacy API"""
        rgb_detect = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)

        # Detect
        results = self.pose_detector.process(rgb_detect)
        has_person = False

        if results.pose_landmarks:
            has_person = True
            # Define custom drawing styles: Blue lines (thickness 4), Cyan circles
            connection_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4)
            landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=5)

            # Draw on the target draw_image (Depth map)
            self.mp_drawing.draw_landmarks(
                draw_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

        return has_person

    def _draw_pose_landmarks_tasks(self, image, landmarks):
        """使用Tasks API在图像上绘制姿态关键点和连接线"""
        try:
            h, w = image.shape[:2]

            # 绘制连接线 - 改为蓝色并加粗
            for connection in self.pose_connections:
                start_idx = connection.start
                end_idx = connection.end
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_landmark = landmarks[start_idx]
                    end_landmark = landmarks[end_idx]

                    if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                        start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                        end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                        cv2.line(image, start_point, end_point, (255, 0, 0), 4)  # BGR: Blue, Thickness: 4

            # 绘制关键点 - 改为青色
            for landmark in landmarks:
                if landmark.visibility > 0.5:  # 只绘制可见的关键点
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(image, (x, y), 5, (255, 255, 0), -1)  # BGR: Cyan, Radius: 5

        except Exception as e:
            print(f"Draw pose landmarks (Tasks API) error: {e}")

    def update_loop(self):
        """UI更新循环"""
        while self.running:
            try:
                self.update_display()
                time.sleep(0.1)  # 10 FPS UI更新
            except Exception as e:
                print(f"UI更新错误: {e}")
                break

    def update_display(self):
        """Update four-view display and status alert"""
        try:
            # Update images
            if hasattr(self, 'rgb_data'):
                self.show_rgb_image()

            if hasattr(self, 'ir_data'):
                self.show_ir_image()

            if hasattr(self, 'depth_data'):
                self.show_depth_image()

            # Update Status Alert (bottom-right)
            self.update_status_alert()

        except Exception as e:
            print(f"Update display error: {e}")

    def update_status_alert(self):
        """Update the status alert canvas based on person detection"""
        try:
            # Update canvas size if needed (on first run or resize)
            width = self.status_canvas.winfo_width()
            height = self.status_canvas.winfo_height()
            if width <= 1: width = 480
            if height <= 1: height = 360

            # Toggle flash state
            self.alert_flash_state = not self.alert_flash_state

            if self.person_detected:
                # PERSON DETECTED: FLASH RED/DARK RED
                bg_color = "red" if self.alert_flash_state else "#8B0000"
                text = "WARNING:\nPERSON DETECTED"
            else:
                # NO PERSON: STEADY GREEN
                bg_color = "green"
                text = "SAFE: NO PERSON"

            # Update canvas
            self.status_canvas.config(bg=bg_color)
            self.status_canvas.itemconfig(self.status_text, text=text)
            self.status_canvas.coords(self.status_text, width//2, height//2)

        except Exception as e:
            print(f"Update status alert error: {e}")


    def show_depth_image(self):
        """Display depth image with skeleton overlay detected from RGB"""
        try:
            if not hasattr(self, 'depth_data'):
                return

            # 1. Normalize depth to 8-bit
            depth_8bit = normalize_depth_to_8bit(self.depth_data, self.depth_max)

            # 2. Apply color map
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_RAINBOW)

            # 3. USE RGB DATA FOR DETECTION (Much higher accuracy)
            # We detect on rgb_data, but draw the skeleton on depth_colored
            if self.mediapipe_available and self.pose_detector and hasattr(self, 'rgb_data'):
                self.process_pose_detection(self.rgb_data, depth_colored)

            # 4. Convert BGR to RGB for PIL display
            depth_display_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(depth_display_rgb)

            # 5. Resize to fit display area
            display_width = 480  # Adjusted for four-view layout
            display_height = 360
            pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

            # 6. Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # 7. Update label
            self.depth_image_label.config(image=photo)
            self.depth_image_label.image = photo

        except Exception as e:
            print(f"Error displaying depth image: {e}")

    def show_rgb_image(self):
        """Display RGB image"""
        try:
            if not hasattr(self, 'rgb_data'):
                return

            # rgb_data is already BGR, convert to RGB for PIL
            rgb_pil = Image.fromarray(cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2RGB))

            # Resize
            display_width = 480
            display_height = 360
            rgb_pil = rgb_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)

            # PhotoImage
            photo = ImageTk.PhotoImage(rgb_pil)

            # Update
            self.rgb_image_label.config(image=photo)
            self.rgb_image_label.image = photo

        except Exception as e:
            print(f"Error displaying RGB image: {e}")

    def show_ir_image(self):
        """Display IR image"""
        try:
            if not hasattr(self, 'ir_data'):
                return

            # Convert to PIL
            ir_pil = Image.fromarray(self.ir_data)

            # Resize
            display_width = 480
            display_height = 360
            ir_pil = ir_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)

            # PhotoImage
            photo = ImageTk.PhotoImage(ir_pil)

            # Update
            self.ir_image_label.config(image=photo)
            self.ir_image_label.image = photo

        except Exception as e:
            print(f"Error displaying IR image: {e}")

    def show_skeleton_image(self):
        """Deprecated: skeleton now shown on depth view"""
        pass

    def save_screenshot(self):
        """Save screenshots of all views"""
        try:
            if not self.connected:
                messagebox.showwarning("Warning", "Please connect camera first")
                return

            # Create save directory
            save_dir = os.path.join(os.path.dirname(__file__), "screenshots")
            os.makedirs(save_dir, exist_ok=True)

            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []

            # Save RGB
            if hasattr(self, 'rgb_data'):
                filename = f"rgb_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, self.rgb_data)
                saved_files.append(f"RGB: {filepath}")

            # Save IR
            if hasattr(self, 'ir_data'):
                filename = f"ir_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, self.ir_data)
                saved_files.append(f"IR: {filepath}")

            # Save Depth (with Skeleton if detected)
            if hasattr(self, 'depth_data'):
                # Normalize depth
                depth_8bit = normalize_depth_to_8bit(self.depth_data, self.depth_max)
                depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_RAINBOW)

                # If person detected, the current 'depth_colored' in memory doesn't have skeleton
                # (since process_pose_detection draws on a local copy used for display).
                # To save what's on screen, we'd need to re-run detection or use the displayed buffer.
                # For simplicity, we save the raw depth visualization.
                filename = f"depth_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, depth_colored)
                saved_files.append(f"Depth: {filepath}")

            if saved_files:
                messagebox.showinfo("Save Successful", f"Screenshots saved:\n" + "\n".join(saved_files))
            else:
                messagebox.showwarning("Warning", "No images available to save")

        except Exception as e:
            messagebox.showerror("Save Failed", f"Screenshot save failed: {str(e)}")

    def clear_display(self):
        """Clear all displays"""
        # Clear images
        self.rgb_image_label.config(image='')
        self.ir_image_label.config(image='')
        self.depth_image_label.config(image='')

        # Reset status alert
        self.person_detected = False
        self.status_canvas.config(bg="gray")
        self.status_canvas.itemconfig(self.status_text, text="DISCONNECTED")

    def on_closing(self):
        """窗口关闭处理"""
        self.running = False
        if self.connected:
            self.disconnect_camera()
        # 清理MediaPipe资源
        if self.pose_detector:
            try:
                if self.mediapipe_mode == "TASKS_API":
                    self.pose_detector.close()
                elif self.mediapipe_mode == "LEGACY_API":
                    self.pose_detector.close()
            except:
                pass
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MultiViewDepthCameraDemo(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
