#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度相机GUI演示程序
支持显示Depth、3D点云、RGB、IR图像
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# 尝试设置matplotlib后端
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib unavailable or TkAgg backend not supported: {e}")
    MATPLOTLIB_AVAILABLE = False

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

class DepthCameraDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Camera Demo")
        self.root.geometry("1200x800")

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

        # 显示模式
        self.display_modes = ["Depth", "3D Point Cloud", "RGB", "IR"]
        self.current_mode = "Depth"

        # 线程控制
        self.running = False
        self.camera_thread = None
        self.update_thread = None

        # 图像显示变量
        self.current_image = None
        self.image_label = None

        # 3D点云显示变量
        self.pointcloud_figure = None
        self.pointcloud_canvas = None
        self.pointcloud_ax = None

        # 状态变量
        self.connected = False

        # 保存相关
        import datetime

        # 创建GUI界面
        self.create_gui()

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

        # Display Mode Selection
        mode_frame = ttk.LabelFrame(control_frame, text="Display Mode", padding="5")
        mode_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))

        self.mode_var = tk.StringVar(value=self.current_mode)
        for i, mode in enumerate(self.display_modes):
            ttk.Radiobutton(mode_frame, text=mode, variable=self.mode_var,
                          value=mode, command=self.change_display_mode).grid(row=i, column=0, sticky=tk.W, pady=2)

        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))

        self.save_btn = ttk.Button(button_frame, text="Save Screenshot", command=self.save_screenshot, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=0, padx=(0, 5))

        self.clear_btn = ttk.Button(button_frame, text="Clear Display", command=self.clear_display)
        self.clear_btn.grid(row=0, column=1)

        # Status Display
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")
        status_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(status_frame, text="Not Connected")
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        # Display Area
        display_frame = ttk.LabelFrame(main_frame, text="Display Area", padding="5")
        display_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 创建图像显示标签（用于2D图像）
        self.image_label = ttk.Label(display_frame)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 创建3D点云显示区域
        self.create_pointcloud_display(display_frame)

        # 配置显示区域的网格权重
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

    def create_pointcloud_display(self, parent):
        """创建3D点云显示区域"""
        if not MATPLOTLIB_AVAILABLE:
            # 如果matplotlib不可用，创建一个占位符标签
            self.pointcloud_placeholder = ttk.Label(parent, text="3D Point Cloud display requires matplotlib support\nPlease install matplotlib and ensure GUI environment")
            self.pointcloud_placeholder.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.pointcloud_canvas = None
            return

        try:
            # 创建matplotlib图形
            self.pointcloud_figure = Figure(figsize=(8, 6), dpi=100)
            self.pointcloud_ax = self.pointcloud_figure.add_subplot(111, projection='3d')
            self.pointcloud_ax.set_title("3D Point Cloud")
            self.pointcloud_ax.set_xlabel("X (毫米)")
            self.pointcloud_ax.set_ylabel("Y (毫米)")
            self.pointcloud_ax.set_zlabel("Z (毫米)")

            # 创建canvas
            self.pointcloud_canvas = FigureCanvasTkAgg(self.pointcloud_figure, master=parent)
            self.pointcloud_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # 初始时隐藏点云显示
            self.pointcloud_canvas.get_tk_widget().grid_remove()
        except Exception as e:
            print(f"创建3D点云显示失败: {e}")
            # 创建错误占位符
            self.pointcloud_placeholder = ttk.Label(parent, text=f"3D Point Cloud display initialization failed:\n{str(e)}")
            self.pointcloud_placeholder.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.pointcloud_canvas = None

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
        """处理深度帧"""
        try:
            # 保存完整的frame对象，用于点云转换
            self.depth_frame = frame

            # 转换深度数据为numpy数组
            depth_array = convert_depthframe_to_array(frame)

            # 保存深度数据用于显示
            self.depth_data = depth_array
            self.depth_frame_info = {
                'width': frame.width,
                'height': frame.height
            }

        except Exception as e:
            print(f"Process depth frame error: {e}")

    def process_ir_frame(self, frame):
        """处理IR帧"""
        try:
            # 转换IR数据为numpy数组
            ir_array = np.ctypeslib.as_array(frame.pFrameData, (1, frame.dataLen))
            ir_array.dtype = np.uint8
            ir_array = ir_array.reshape((frame.height, frame.width))

            # 保存IR数据用于显示
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

            # 保存RGB数据用于显示
            self.rgb_data = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"Process RGB frame error: {e}")

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
        """更新显示"""
        try:
            if self.current_mode == "Depth" and hasattr(self, 'depth_data'):
                self.show_depth_image()
            elif self.current_mode == "3D Point Cloud" and hasattr(self, 'depth_data'):
                self.show_pointcloud()
            elif self.current_mode == "RGB" and hasattr(self, 'rgb_data'):
                self.show_rgb_image()
            elif self.current_mode == "IR" and hasattr(self, 'ir_data'):
                self.show_ir_image()
        except Exception as e:
            print(f"Update display error: {e}")

    def change_display_mode(self):
        """切换显示模式"""
        self.current_mode = self.mode_var.get()
        print(f"Switching to display mode: {self.current_mode}")
        self.clear_display()

        if self.current_mode == "3D Point Cloud":
            # 显示3D点云
            self.image_label.grid_remove()
            if self.pointcloud_canvas:
                self.pointcloud_canvas.get_tk_widget().grid()
                print("Showing 3D point cloud canvas")
            elif hasattr(self, 'pointcloud_placeholder'):
                self.pointcloud_placeholder.grid()
                print("Showing 3D point cloud placeholder")
        else:
            # 显示2D图像
            if self.pointcloud_canvas:
                self.pointcloud_canvas.get_tk_widget().grid_remove()
            if hasattr(self, 'pointcloud_placeholder'):
                self.pointcloud_placeholder.grid_remove()
            self.image_label.grid()
            print("Showing 2D image display")

    def show_depth_image(self):
        """显示深度图像"""
        try:
            if not hasattr(self, 'depth_data'):
                return

            # 归一化深度数据为8位图像
            depth_8bit = normalize_depth_to_8bit(self.depth_data, self.depth_max)

            # 应用颜色映射
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_RAINBOW)

            # 转换为PIL图像
            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(depth_rgb)

            # 调整大小
            pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # 更新显示
            self.image_label.config(image=photo)
            self.image_label.image = photo

        except Exception as e:
            print(f"显示深度图像错误: {e}")

    def show_pointcloud(self):
        """显示3D点云"""
        if not MATPLOTLIB_AVAILABLE or not self.pointcloud_canvas:
            print("Matplotlib not available or canvas not ready")
            return

        try:
            if not hasattr(self, 'depth_frame'):
                print("No depth frame available")
                return

            if not self.camera:
                print("Camera not available")
                return

            print(f"Converting depth frame to point cloud: {self.depth_frame.width}x{self.depth_frame.height}")

            # 转换深度帧为点云
            ret, pointlist = self.camera.scConvertDepthFrameToPointCloudVector(self.depth_frame)
            if ret != 0:
                print(f"Point cloud conversion failed: {ret}")
                return

            print(f"Point cloud conversion successful, processing {self.depth_frame.width * self.depth_frame.height} points")

            # 提取有效的点云数据
            points = []
            valid_count = 0
            for i in range(self.depth_frame.width * self.depth_frame.height):
                if pointlist[i].z != 0 and pointlist[i].z != 65535:
                    points.append([pointlist[i].x, pointlist[i].y, pointlist[i].z])
                    valid_count += 1
                    if valid_count >= 50000:  # 限制最大点数
                        break

            print(f"Found {len(points)} valid points")

            if not points:
                print("No valid points found")
                return

            points = np.array(points)

            # 随机采样点云以提高性能（可选）
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points = points[indices]

            # 更新3D显示
            self.pointcloud_ax.clear()
            self.pointcloud_ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                     c=points[:, 2], cmap='viridis', s=1, alpha=0.8)
            self.pointcloud_ax.set_title("3D Point Cloud")
            self.pointcloud_ax.set_xlabel("X (毫米)")
            self.pointcloud_ax.set_ylabel("Y (毫米)")
            self.pointcloud_ax.set_zlabel("Z (毫米)")

            # 设置合适的视角
            self.pointcloud_ax.view_init(elev=20, azim=45)

            # 设置轴的范围
            if len(points) > 0:
                x_range = points[:, 0].max() - points[:, 0].min()
                y_range = points[:, 1].max() - points[:, 1].min()
                z_range = points[:, 2].max() - points[:, 2].min()
                max_range = max(x_range, y_range, z_range)
                mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
                mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
                mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
                self.pointcloud_ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
                self.pointcloud_ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
                self.pointcloud_ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

            # 刷新canvas
            self.pointcloud_canvas.draw()

        except Exception as e:
            print(f"显示点云错误: {e}")

    def show_rgb_image(self):
        """显示RGB图像"""
        try:
            if not hasattr(self, 'rgb_data'):
                return

            # 转换为PIL图像
            rgb_pil = Image.fromarray(cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2RGB))

            # 调整大小
            rgb_pil = rgb_pil.resize((640, 480), Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(rgb_pil)

            # 更新显示
            self.image_label.config(image=photo)
            self.image_label.image = photo

        except Exception as e:
            print(f"显示RGB图像错误: {e}")

    def show_ir_image(self):
        """显示IR图像"""
        try:
            if not hasattr(self, 'ir_data'):
                return

            # 转换为PIL图像
            ir_pil = Image.fromarray(self.ir_data)

            # 调整大小
            ir_pil = ir_pil.resize((640, 480), Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(ir_pil)

            # 更新显示
            self.image_label.config(image=photo)
            self.image_label.image = photo

        except Exception as e:
            print(f"显示IR图像错误: {e}")

    def save_screenshot(self):
        """保存当前显示的截图"""
        try:
            if not self.connected:
                messagebox.showwarning("Warning", "Please connect camera first")
                return

            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(__file__), "screenshots")
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_name = self.current_mode.replace(" ", "_")

            if self.current_mode == "3D Point Cloud":
                # 保存点云截图
                filename = f"pointcloud_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                self.pointcloud_figure.savefig(filepath, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Save Successful", f"Point cloud screenshot saved to: {filepath}")
            else:
                # 保存2D图像
                if hasattr(self, 'current_image') and self.current_image:
                    filename = f"{mode_name}_{timestamp}.png"
                    filepath = os.path.join(save_dir, filename)
                    self.current_image.save(filepath)
                    messagebox.showinfo("Save Successful", f"Image saved to: {filepath}")
                else:
                    messagebox.showwarning("Warning", "No image available to save")

        except Exception as e:
            messagebox.showerror("Save Failed", f"Screenshot save failed: {str(e)}")

    def clear_display(self):
        """清空显示"""
        # 清空图像显示
        self.image_label.config(image='')

        # 清空点云显示
        if MATPLOTLIB_AVAILABLE and self.pointcloud_canvas and self.pointcloud_ax:
            self.pointcloud_ax.clear()
            self.pointcloud_canvas.draw()

    def on_closing(self):
        """窗口关闭处理"""
        self.running = False
        if self.connected:
            self.disconnect_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = DepthCameraDemo(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
