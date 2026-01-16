#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Camera Tutorial Version 2: Direct Depth Image Display
Demonstrates how to acquire and display depth images
"""
import sys
import os
import cv2
import numpy as np
import time

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
from utils import convert_depthframe_to_array, normalize_depth_to_8bit

def detect_and_connect_camera():
    """Detect and connect camera"""
    try:
        # Import SDK
        from sdk_loader import load_scepter_sdk
        from camera_manager import initialize_camera as _initialize_camera, open_and_start_stream as _open_and_start_stream

        # Load SDK
        print("Loading SDK...")
        sdk_loaded, sdk_base_path, ScepterTofCam, ScSensorType, ScFrameType, ScWorkMode, ScExposureControlMode, ScTimeFilterParams, ScConfidenceFilterParams, ScFlyingPixelFilterParams = load_scepter_sdk()

        if not sdk_loaded:
            print("❌ SDK loading failed")
            return None, None, None, None, None

        # Initialize camera
        camera, device_info, device_already_open = _initialize_camera(ScepterTofCam)
        if not camera or not device_info:
            print("❌ Camera initialization failed")
            return None, None, None, None, None

        # Start stream (simplified version, only enable depth)
        depth_max, camera_param, camera_actual_fps, actual_rgb_resolution = _open_and_start_stream(
            camera, device_info, ScSensorType, ScWorkMode,
            enable_rgb=False, rgb_resolution=(640, 480), enable_ir=False, target_fps=30,
            ScTimeFilterParams=ScTimeFilterParams, ScConfidenceFilterParams=ScConfidenceFilterParams,
            ScFlyingPixelFilterParams=ScFlyingPixelFilterParams, ScExposureControlMode=ScExposureControlMode,
            device_already_open=device_already_open
        )

        print("✅ Camera connected successfully")
        print(f"   FPS: {camera_actual_fps}")
        print(f"   Maximum depth value: {depth_max}")

        return camera, ScFrameType, depth_max, camera_param, camera_actual_fps

    except Exception as e:
        print(f"❌ Camera connection failed: {e}")
        return None, None, None, None, None

def process_depth_frame(frame, depth_max):
    """Process depth frame and convert to displayable image"""
    try:
        # Convert depth data to numpy array
        depth_array = convert_depthframe_to_array(frame)

        # Normalize depth data to 8-bit image
        depth_8bit = normalize_depth_to_8bit(depth_array, depth_max)

        # Apply color map (rainbow)
        depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_RAINBOW)

        return depth_colored

    except Exception as e:
        print(f"Depth frame processing error: {e}")
        return None

def display_depth_stream(camera, ScFrameType, depth_max):
    """Display depth image stream"""
    print("Starting depth image stream display...")
    print("Press 'q' or 'ESC' to exit")

    window_name = "Depth Image"

    try:
        while True:
            # Check if frame is ready
            ret, frameready = camera.scGetFrameReady(1200)
            if ret != 0:
                continue

            # Get depth frame
            if frameready.depth:
                ret, depth_frame = camera.scGetFrame(ScFrameType.SC_DEPTH_FRAME)
                if ret == 0:
                    # Process and display depth image
                    depth_image = process_depth_frame(depth_frame, depth_max)
                    if depth_image is not None:
                        cv2.imshow(window_name, depth_image)

            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("Interrupt signal received")
    except Exception as e:
        print(f"Error during display: {e}")
    finally:
        cv2.destroyAllWindows()

def disconnect_camera(camera):
    """Disconnect camera"""
    try:
        print("Disconnecting camera...")
        if camera:
            camera.scStopStream()
            camera.scCloseDevice()
            print("✅ Camera disconnected")
    except Exception as e:
        print(f"❌ Error disconnecting camera: {e}")

def main():
    """Main function"""
    print("=== Depth Camera Tutorial Version 2: Direct Depth Image Display ===")
    print()

    # Step 1: Connect camera
    print("Step 1: Connect camera")
    camera, ScFrameType, depth_max, camera_param, camera_actual_fps = detect_and_connect_camera()

    if not camera:
        print("Cannot proceed, please check camera connection")
        return

    print()

    # Step 2: Display depth image
    print("Step 2: Display depth image")
    try:
        display_depth_stream(camera, ScFrameType, depth_max)
    except Exception as e:
        print(f"Error displaying depth image: {e}")

    print()

    # Step 3: Disconnect
    print("Step 3: Disconnect")
    disconnect_camera(camera)

    print()
    print("=== Demo Completed ===")

if __name__ == "__main__":
    main()