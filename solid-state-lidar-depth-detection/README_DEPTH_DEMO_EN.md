# Depth Camera Demo Program - Educational Version

This directory contains multiple educational versions of the depth camera demo program, progressing from basic to complete implementations.

## Version Descriptions

### Version 1: camera_detection.py - Camera Detection and Connection
**Learning Objective**: Understand how to detect camera devices and establish connections

**Features**:
- Detect depth camera devices on the network
- Display camera information (serial number, IP address, connection status)
- Establish camera connection and start data stream
- Disconnect camera connection

**Teaching Focus**:
- SDK initialization
- Device enumeration
- Camera connection process
- Basic error handling

### Version 2: depth_display.py - Direct Depth Image Display
**Learning Objective**: Understand how to acquire and display depth images

**Features**:
- Camera connection (based on Version 1)
- Acquire depth frame data
- Convert depth data to visualizable images
- Real-time display of depth images using OpenCV window
- Color depth map display (using pseudo-color mapping)

**Teaching Focus**:
- Frame data acquisition
- Depth data processing
- Image visualization
- Real-time display loop

### Version 3: gui_selector.py - GUI Selector
**Learning Objective**: Understand GUI interface design and multi-mode display

**Features**:
- Graphical User Interface (Tkinter)
- Camera connection control
- Multi-display mode selection (Depth, RGB, IR)
- Real-time image display
- Status information display

**Teaching Focus**:
- GUI application structure
- Event-driven programming
- Multi-threading (camera data acquisition and UI updates)
- User interface design

### Version 4: depth_demo_full.py - Complete Version
**Learning Objective**: Complete depth camera application development

**Features**:
- All features from Version 3
- 3D point cloud display (using matplotlib)
- Screenshot saving functionality
- Complete state management
- Error handling and resource cleanup

**Teaching Focus**:
- Complex GUI design
- 3D data visualization
- File I/O operations
- Complete application architecture

## System Requirements

### Dependencies
```bash
pip install opencv-python numpy pillow matplotlib
```

### System Requirements
- Linux (Ubuntu/Armbian recommended)
- Depth camera device (network connected)
- GUI environment (X11 support)

### SDK Requirements
- Scepter SDK properly installed
- Library paths configured (LD_LIBRARY_PATH)

## Usage Instructions

### Preparation
1. Ensure depth camera is connected to the network
2. Ensure camera power is turned on
3. Verify SDK path is correct before running

### Running Steps
```bash
# Version 1: Camera Detection
python camera_detection.py

# Version 2: Depth Image Display
python depth_display.py

# Version 3: GUI Selector
python gui_selector.py

# Version 4: Complete Version
python depth_demo_full.py
```

## Code Structure Explanation

### Core Modules (located in ad-depth directory)
- `sdk_loader.py`: SDK loading and initialization
- `camera_manager.py`: Camera connection and management
- `utils.py`: Utility functions (image processing, etc.)
- `frame_processor.py`: Frame data processing

### Version Files
- `camera_detection.py`: Version 1 - Basic camera connection
- `depth_display.py`: Version 2 - Depth image display
- `gui_selector.py`: Version 3 - GUI multi-mode selection
- `depth_demo_full.py`: Version 4 - Complete feature version

## Recommended Learning Path

1. **Start with Version 1**: Understand basic camera connection concepts
2. **Version 2**: Learn image data processing and display
3. **Version 3**: Master GUI programming and multi-threading
4. **Version 4**: Comprehensive application, learn complete application development

Each version includes detailed comments and error handling, suitable for educational use.

## Common Issues

### SDK Loading Failure
- Check if SDK path is correct
- Confirm LD_LIBRARY_PATH settings
- Check if Python path includes SDK directory

### Camera Connection Failure
- Check camera network connection
- Confirm camera power status
- Check firewall settings

### GUI Display Issues
- Confirm X11 environment is available
- Check matplotlib backend settings
- Verify tkinter is properly installed

### 3D Point Cloud Not Displaying
- Confirm matplotlib is installed
- Check OpenGL support
- Check if terminal has GUI permissions

## Technical Highlights

- **Multi-threading Programming**: Separation of camera data acquisition and UI updates
- **Image Processing**: Conversion of depth data to visual images
- **3D Visualization**: matplotlib 3D scatter plot display
- **Network Communication**: Camera connection via TCP/IP
- **GUI Design**: Tkinter event-driven interface

## Extension Directions

- Add more image processing algorithms
- Implement depth data saving functionality
- Add camera parameter adjustment interface
- Support simultaneous display of multiple cameras
- Add video recording functionality
