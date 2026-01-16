#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Camera Tutorial Version 1: Camera Detection and Connection
Demonstrates how to detect camera devices and establish connection
"""
import sys
import os

# Add related_code directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ad_depth_dir = os.path.join(current_dir, "related_code")
if ad_depth_dir not in sys.path:
    sys.path.insert(0, ad_depth_dir)

# Set SDK path environment variable
sdk_python_path = os.path.join(ad_depth_dir, "scepter_sdk", "ScepterSDK-master", "MultilanguageSDK", "Python")
if os.path.exists(sdk_python_path):
    os.environ['SCEPTER_SDK_PATH'] = sdk_python_path
    print(f"Set SDK path: {sdk_python_path}")
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
        print(f"Set library path: {lib_path}")

def detect_camera():
    """Detect camera devices"""
    try:
        # Import SDK
        from sdk_loader import load_scepter_sdk

        # Load SDK
        print("Loading SDK...")
        sdk_loaded, sdk_base_path, ScepterTofCam, ScSensorType, ScFrameType, ScWorkMode, ScExposureControlMode, ScTimeFilterParams, ScConfidenceFilterParams, ScFlyingPixelFilterParams = load_scepter_sdk()

        if not sdk_loaded:
            print("❌ SDK loading failed")
            return None, None

        print("✅ SDK loaded successfully")

        # Create camera instance
        camera = ScepterTofCam()

        # Detect number of devices
        print("Detecting camera devices...")
        camera_count = camera.scGetDeviceCount(3000)
        print(f"Found {camera_count} camera device(s)")

        if camera_count <= 0:
            print("❌ No camera devices found")
            return None, None

        # Get device information
        ret, device_infolist = camera.scGetDeviceInfoList(camera_count)
        if ret != 0:
            print(f"❌ Failed to get device information: {ret}")
            return None, None

        device_info = device_infolist[0]
        print("✅ Found camera device:")
        print(f"   Serial number: {device_info.serialNumber}")
        print(f"   IP address: {device_info.ip}")
        print(f"   Connection status: {device_info.status}")

        return camera, device_info

    except Exception as e:
        print(f"❌ Camera detection failed: {e}")
        return None, None

def connect_camera(camera, device_info):
    """Connect to camera"""
    try:
        print("Connecting to camera...")

        # Open device
        ret = camera.scOpenDeviceBySN(device_info.serialNumber)
        if ret != 0:
            print(f"❌ Failed to open device: {ret}")
            return False

        print("✅ Device opened successfully")

        # Start stream
        ret = camera.scStartStream()
        if ret != 0:
            print(f"❌ Failed to start stream: {ret}")
            camera.scCloseDevice()
            return False

        print("✅ Camera stream started successfully")
        print("🎉 Camera connection completed!")

        return True

    except Exception as e:
        print(f"❌ Camera connection failed: {e}")
        return False

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
    print("=== Depth Camera Tutorial Version 1: Camera Detection and Connection ===")
    print()

    # Step 1: Detect camera
    print("Step 1: Detect camera devices")
    camera, device_info = detect_camera()

    if not camera or not device_info:
        print("Cannot proceed, please check camera connection")
        return

    print()

    # Step 2: Connect to camera
    print("Step 2: Connect to camera")
    connected = connect_camera(camera, device_info)

    if connected:
        print()
        print("=== Demo Completed ===")
        print("Camera successfully connected, ready for next learning step")
        print()

        # Wait for user input
        input("Press Enter to disconnect and exit...")

        # Disconnect
        disconnect_camera(camera)
    else:
        print("Camera connection failed")

if __name__ == "__main__":
    main()