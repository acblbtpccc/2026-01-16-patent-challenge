#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Camera Tutorial Version 4: Complete Version
Complete implementation with all features: camera connection, multiple display modes, 3D point cloud, screenshot saving, etc.
"""
import sys
import os
import cv2
import numpy as np
import datetime
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Try to set matplotlib backend
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib unavailable: {e}")
    MATPLOTLIB_AVAILABLE = False

# Add related_code directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ad_depth_dir = os.path.join(current_dir, "related_code")
if ad_depth_dir not in sys.path:
    sys.path.insert(0, ad_depth_dir)

# Set SDK path environment variable
sdk_python_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "MultilanguageSDK", "Python")
if os.path.exists(sdk_python_path):
    os.environ['SCEPTER_SDK_PATH'] = sdk_python_path
else:
    print(f"SDK path does not exist: {sdk_python_path}")

# Set library path
sdk_base_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "BaseSDK")
if os.path.exists(sdk_base_path):
    lib_path = os.path.join(sdk_base_path, "AArch64", "Lib")
    if os.path.exists(lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = lib_path

# Import utility functions
try:
    from sdk_loader import load_scepter_sdk
    from camera_manager import initialize_camera as _initialize_camera, open_and_start_stream as _open_and_start_stream
    from utils import convert_depthframe_to_array, normalize_depth_to_8bit
    from frame_processor import process_frame_from_array, visualize_pointcloud
    SDK_AVAILABLE = True
    print("SDK modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    SDK_AVAILABLE = False

class DepthCameraDemo:
    """Depth Camera Demo Program Main Class"""

    def __init__(self, root):
        """Initialize"""
        self.root = root
        self.root.title("Depth Camera Demo - Full Version")
        self.root.geometry("1200x800")

        # Check SDK availability
        if not SDK_AVAILABLE:
            messagebox.showerror("Error", "SDK modules not available")
            return

        # Camera related variables
        self.camera = None
        self.device_info = None
        self.depth_max = None
        self.camera_param = None
        self.ScFrameType = None
        self.camera_actual_fps = None

        # Display modes
        self.display_modes = ["Depth", "3D Point Cloud", "RGB", "IR"]
        self.current_mode = "Depth"

        # Thread control
        self.running = False
        self.camera_thread = None
        self.update_thread = None

        # GUI components
        self.image_label = None
        self.pointcloud_canvas = None
        self.status_label = None

        # Create GUI
        self.create_gui()

    def create_gui(self):
        """Create GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))

        # Connect/Disconnect buttons
        self.connect_btn = ttk.Button(control_frame, text="Connect Camera", command=self.connect_camera)
        self.connect_btn.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        self.disconnect_btn = ttk.Button(control_frame, text="Disconnect", command=self.disconnect_camera, state=tk.DISABLED)
        self.disconnect_btn.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))

        # Display mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Display Mode", padding="5")
        mode_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))

        self.mode_var = tk.StringVar(value=self.current_mode)
        for i, mode in enumerate(self.display_modes):
            ttk.Radiobutton(mode_frame, text=mode, variable=self.mode_var,
                          value=mode, command=self.change_display_mode).grid(row=i, column=0, sticky=tk.W, pady=2)

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))

        self.save_btn = ttk.Button(button_frame, text="Save Screenshot", command=self.save_screenshot, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=0, padx=(0, 5))

        self.clear_btn = ttk.Button(button_frame, text="Clear Display", command=self.clear_display)
        self.clear_btn.grid(row=0, column=1)

        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")
        status_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(status_frame, text="Not Connected")
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        # Display area
        display_frame = ttk.LabelFrame(main_frame, text="Display Area", padding="5")
        display_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create image display
        self.image_label = ttk.Label(display_frame)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create 3D point cloud display
        self.create_pointcloud_display(display_frame)

        # Configure display area
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

    def create_pointcloud_display(self, parent):
        """Create 3D point cloud display area"""
        if not MATPLOTLIB_AVAILABLE:
            self.pointcloud_placeholder = ttk.Label(parent, text="3D Point Cloud requires matplotlib")
            return

        # Create matplotlib figure
        self.pointcloud_figure = Figure(figsize=(8, 6), dpi=100)
        self.pointcloud_ax = self.pointcloud_figure.add_subplot(111, projection='3d')
        self.pointcloud_ax.set_title("3D Point Cloud")
        self.pointcloud_ax.set_xlabel("X (mm)")
        self.pointcloud_ax.set_ylabel("Y (mm)")
        self.pointcloud_ax.set_zlabel("Z (mm)")

        # Create canvas
        self.pointcloud_canvas = FigureCanvasTkAgg(self.pointcloud_figure, master=parent)
        self.pointcloud_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.pointcloud_canvas.get_tk_widget().grid_remove()

    def connect_camera(self):
        """Connect camera"""
        try:
            self.status_label.config(text="Connecting...")
            self.connect_btn.config(state=tk.DISABLED)

            # Load SDK and initialize camera
            sdk_loaded, sdk_base_path, ScepterTofCam, ScSensorType, ScFrameType, ScWorkMode, ScExposureControlMode, ScTimeFilterParams, ScConfidenceFilterParams, ScFlyingPixelFilterParams = load_scepter_sdk()
            if not sdk_loaded:
                raise RuntimeError("SDK load failed")

            self.ScFrameType = ScFrameType

            # Initialize camera
            camera, device_info, device_already_open = _initialize_camera(ScepterTofCam)
            if not camera or not device_info:
                raise RuntimeError("Camera initialization failed")

            # Start stream
            depth_max, camera_param, camera_actual_fps, actual_rgb_resolution = _open_and_start_stream(
                camera, device_info, ScSensorType, ScWorkMode,
                enable_rgb=True, rgb_resolution=(640, 480), enable_ir=True, target_fps=30,
                ScTimeFilterParams=ScTimeFilterParams, ScConfidenceFilterParams=ScConfidenceFilterParams,
                ScFlyingPixelFilterParams=ScFlyingPixelFilterParams, ScExposureControlMode=ScExposureControlMode,
                device_already_open=device_already_open
            )

            # Save camera information
            self.camera = camera
            self.device_info = device_info
            self.depth_max = depth_max
            self.camera_param = camera_param
            self.camera_actual_fps = camera_actual_fps

            # Update status
            self.status_label.config(text=f"Connected - FPS: {camera_actual_fps}")
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)

            # Start threads
            self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
            self.update_thread.start()

        except Exception as e:
            messagebox.showerror("Connection Failed", f"Camera connection failed: {str(e)}")
            self.status_label.config(text="Connection Failed")
            self.connect_btn.config(state=tk.NORMAL)

    def disconnect_camera(self):
        """Disconnect camera"""
        try:
            self.running = False
            self.status_label.config(text="Disconnecting...")

            # Wait for threads to finish
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2.0)
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2.0)

            # Close camera
            if self.camera:
                self.camera.scStopStream()
                self.camera.scCloseDevice()

            # Reset status
            self.camera = None
            self.status_label.config(text="Disconnected")
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)

            # Clear display
            self.clear_display()

        except Exception as e:
            messagebox.showerror("Disconnect Failed", f"Camera disconnect failed: {str(e)}")

    def camera_loop(self):
        """Camera data acquisition loop"""
        while self.running:
            try:
                ret, frameready = self.camera.scGetFrameReady(1200)
                if ret != 0:
                    continue

                # Get various frame data
                if frameready.depth:
                    ret, frame = self.camera.scGetFrame(self.ScFrameType.SC_DEPTH_FRAME)
                    if ret == 0:
                        self.depth_frame = frame
                        self.depth_data = convert_depthframe_to_array(frame)

                if frameready.ir:
                    ret, frame = self.camera.scGetFrame(self.ScFrameType.SC_IR_FRAME)
                    if ret == 0:
                        ir_array = np.ctypeslib.as_array(frame.pFrameData, (1, frame.dataLen))
                        ir_array.dtype = np.uint8
                        self.ir_data = ir_array.reshape((frame.height, frame.width))

                if frameready.color:
                    ret, frame = self.camera.scGetFrame(self.ScFrameType.SC_COLOR_FRAME)
                    if ret == 0:
                        rgb_array = np.ctypeslib.as_array(frame.pFrameData, (1, frame.width * frame.height * 3))
                        rgb_array.dtype = np.uint8
                        self.rgb_data = cv2.cvtColor(
                            rgb_array.reshape((frame.height, frame.width, 3)),
                            cv2.COLOR_RGB2BGR
                        )

                time.sleep(0.033)

            except Exception as e:
                print(f"Camera loop error: {e}")
                break

    def update_loop(self):
        """UI update loop"""
        while self.running:
            try:
                self.update_display()
                time.sleep(0.1)
            except Exception as e:
                print(f"Update loop error: {e}")
                break

    def update_display(self):
        """Update display"""
        try:
            mode = self.current_mode
            if mode == "Depth" and hasattr(self, 'depth_data'):
                self.show_depth_image()
            elif mode == "3D Point Cloud" and hasattr(self, 'depth_frame'):
                self.show_pointcloud()
            elif mode == "RGB" and hasattr(self, 'rgb_data'):
                self.show_rgb_image()
            elif mode == "IR" and hasattr(self, 'ir_data'):
                self.show_ir_image()
        except Exception as e:
            print(f"Update display error: {e}")

    def show_depth_image(self):
        """Display depth image"""
        try:
            depth_8bit = normalize_depth_to_8bit(self.depth_data, self.depth_max)
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_RAINBOW)

            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(depth_rgb)
            pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

        except Exception as e:
            print(f"Show depth image error: {e}")

    def show_pointcloud(self):
        """Display 3D point cloud"""
        if not MATPLOTLIB_AVAILABLE or not self.pointcloud_canvas:
            return

        try:
            ret, pointlist = self.camera.scConvertDepthFrameToPointCloudVector(self.depth_frame)
            if ret != 0:
                print(f"Point cloud conversion failed: {ret}")
                return

            points = []
            for i in range(self.depth_frame.width * self.depth_frame.height):
                if pointlist[i].z != 0 and pointlist[i].z != 65535:
                    points.append([pointlist[i].x, pointlist[i].y, pointlist[i].z])

            if not points:
                return

            points = np.array(points)
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points = points[indices]

            self.pointcloud_ax.clear()
            self.pointcloud_ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                     c=points[:, 2], cmap='viridis', s=1, alpha=0.8)
            self.pointcloud_ax.set_title("3D Point Cloud")
            self.pointcloud_ax.set_xlabel("X (mm)")
            self.pointcloud_ax.set_ylabel("Y (mm)")
            self.pointcloud_ax.set_zlabel("Z (mm)")

            self.pointcloud_canvas.draw()

        except Exception as e:
            print(f"Show point cloud error: {e}")

    def show_rgb_image(self):
        """Display RGB image"""
        try:
            rgb_pil = Image.fromarray(cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2RGB))
            rgb_pil = rgb_pil.resize((640, 480), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(rgb_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo

        except Exception as e:
            print(f"Show RGB image error: {e}")

    def show_ir_image(self):
        """Display IR image"""
        try:
            ir_pil = Image.fromarray(self.ir_data)
            ir_pil = ir_pil.resize((640, 480), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(ir_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo

        except Exception as e:
            print(f"Show IR image error: {e}")

    def change_display_mode(self):
        """Switch display mode"""
        self.current_mode = self.mode_var.get()
        self.clear_display()

        if self.current_mode == "3D Point Cloud":
            self.image_label.grid_remove()
            if self.pointcloud_canvas:
                self.pointcloud_canvas.get_tk_widget().grid()
        else:
            if self.pointcloud_canvas:
                self.pointcloud_canvas.get_tk_widget().grid_remove()
            self.image_label.grid()

    def save_screenshot(self):
        """Save screenshot"""
        try:
            if not self.camera:
                messagebox.showwarning("Warning", "Please connect camera first")
                return

            save_dir = os.path.join(os.path.dirname(__file__), "screenshots")
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_name = self.current_mode.replace(" ", "_")

            if self.current_mode == "3D Point Cloud":
                filename = f"pointcloud_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                self.pointcloud_figure.savefig(filepath, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Save Successful", f"Point cloud saved to: {filepath}")
            else:
                # Saving 2D images requires additional logic, simplified here
                messagebox.showinfo("Save", f"2D image save not implemented for {self.current_mode}")

        except Exception as e:
            messagebox.showerror("Save Failed", f"Screenshot save failed: {str(e)}")

    def clear_display(self):
        """Clear display"""
        self.image_label.config(image='')
        if MATPLOTLIB_AVAILABLE and self.pointcloud_canvas and self.pointcloud_ax:
            self.pointcloud_ax.clear()
            self.pointcloud_canvas.draw()

    def on_closing(self):
        """Window closing handler"""
        self.running = False
        if self.camera:
            self.disconnect_camera()
        self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = DepthCameraDemo(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()