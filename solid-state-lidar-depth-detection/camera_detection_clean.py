#!/usr/bin/env python3
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
ad_depth_dir = os.path.join(current_dir, "related_code")
if ad_depth_dir not in sys.path:
    sys.path.insert(0, ad_depth_dir)

sdk_python_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "MultilanguageSDK", "Python")
if os.path.exists(sdk_python_path):
    os.environ['SCEPTER_SDK_PATH'] = sdk_python_path
else:
    print(f"SDK path not found: {sdk_python_path}")

sdk_base_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "BaseSDK")
if os.path.exists(sdk_base_path):
    lib_path = os.path.join(sdk_base_path, "AArch64", "Lib")
    if os.path.exists(lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path

def detect_camera():q
    try:
        from sdk_loader import load_scepter_sdk
        sdk_loaded, _, ScepterTofCam, _, _, _, _, _, _, _ = load_scepter_sdk()
        if not sdk_loaded:
            return None, None

        camera = ScepterTofCam()
        camera_count = camera.scGetDeviceCount(3000)
        if camera_count <= 0:
            return None, None

        ret, device_infolist = camera.scGetDeviceInfoList(camera_count)
        if ret != 0:
            return None, None

        device_info = device_infolist[0]
        return camera, device_info

    except Exception as e:
        print(f"Camera detection failed: {e}")
        return None, None

def connect_camera(camera, device_info):
    try:
        ret = camera.scOpenDeviceBySN(device_info.serialNumber)
        if ret != 0:
            return False

        ret = camera.scStartStream()
        if ret != 0:
            camera.scCloseDevice()
            return False

        return True

    except Exception as e:
        print(f"Camera connection failed: {e}")
        return False

def disconnect_camera(camera):
    try:
        if camera:
            camera.scStopStream()
            camera.scCloseDevice()
    except Exception as e:
        print(f"Disconnect error: {e}")

def main():
    print("=== Camera Detection Demo ===")

    camera, device_info = detect_camera()
    if not camera:
        print("No camera found")
        return

    connected = connect_camera(camera, device_info)
    if connected:
        print("Camera connected successfully")
        input("Press Enter to disconnect...")
        disconnect_camera(camera)
        print("Camera disconnected")
    else:
        print("Connection failed")

if __name__ == "__main__":
    main()
