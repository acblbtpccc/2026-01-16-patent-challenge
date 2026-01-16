#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
ad_depth_dir = os.path.join(current_dir, "related_code")
if ad_depth_dir not in sys.path:
    sys.path.insert(0, ad_depth_dir)

sdk_python_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "MultilanguageSDK", "Python")
if os.path.exists(sdk_python_path):
    os.environ['SCEPTER_SDK_PATH'] = sdk_python_path

sdk_base_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "BaseSDK")
if os.path.exists(sdk_base_path):
    lib_path = os.path.join(sdk_base_path, "AArch64", "Lib")
    if os.path.exists(lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path

from utils import convert_depthframe_to_array, normalize_depth_to_8bit

def detect_and_connect_camera():
    try:
        from sdk_loader import load_scepter_sdk
        from camera_manager import initialize_camera as _initialize_camera, open_and_start_stream as _open_and_start_stream

        sdk_loaded, _, ScepterTofCam, ScSensorType, ScFrameType, ScWorkMode, ScExposureControlMode, ScTimeFilterParams, ScConfidenceFilterParams, ScFlyingPixelFilterParams = load_scepter_sdk()
        if not sdk_loaded:
            return None, None, None, None, None

        camera, device_info, device_already_open = _initialize_camera(ScepterTofCam)
        if not camera:
            return None, None, None, None, None

        depth_max, camera_param, camera_actual_fps, _ = _open_and_start_stream(
            camera, device_info, ScSensorType, ScWorkMode,
            enable_rgb=False, rgb_resolution=(640, 480), enable_ir=False, target_fps=30,
            ScTimeFilterParams=ScTimeFilterParams, ScConfidenceFilterParams=ScConfidenceFilterParams,
            ScFlyingPixelFilterParams=ScFlyingPixelFilterParams, ScExposureControlMode=ScExposureControlMode,
            device_already_open=device_already_open
        )

        return camera, ScFrameType, depth_max, camera_param, camera_actual_fps

    except Exception as e:
        print(f"Camera setup failed: {e}")
        return None, None, None, None, None

def process_depth_frame(frame, depth_max):
    depth_array = convert_depthframe_to_array(frame)
    depth_8bit = normalize_depth_to_8bit(depth_array, depth_max)
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_RAINBOW)
    return depth_colored

def display_depth_stream(camera, ScFrameType, depth_max):
    print("Press 'q' or ESC to exit")

    try:
        while True:
            ret, frameready = camera.scGetFrameReady(1200)
            if ret != 0:
                continue

            if frameready.depth:
                ret, depth_frame = camera.scGetFrame(ScFrameType.SC_DEPTH_FRAME)
                if ret == 0:
                    depth_image = process_depth_frame(depth_frame, depth_max)
                    if depth_image is not None:
                        cv2.imshow("Depth Image", depth_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            time.sleep(0.033)

    except Exception as e:
        print(f"Display error: {e}")
    finally:
        cv2.destroyAllWindows()

def disconnect_camera(camera):
    try:
        if camera:
            camera.scStopStream()
            camera.scCloseDevice()
    except Exception as e:
        print(f"Disconnect error: {e}")

def main():
    print("=== Depth Display Demo ===")

    camera, ScFrameType, depth_max, _, _ = detect_and_connect_camera()
    if not camera:
        return

    try:
        display_depth_stream(camera, ScFrameType, depth_max)
    except Exception as e:
        print(f"Display failed: {e}")

    disconnect_camera(camera)
    print("Demo completed")

if __name__ == "__main__":
    main()
