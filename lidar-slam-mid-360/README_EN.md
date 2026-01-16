# LiDAR SLAM Demonstration System

## What is SLAM?

SLAM (Simultaneous Localization and Mapping) is the technology of **simultaneous localization and mapping**. In unknown environments, robots or devices need to:

1. **Determine their own position** (Localization)
2. **Build a map of the surrounding environment** (Mapping)

These two processes occur simultaneously and are interdependent, which is the core concept of SLAM.

## System Components

This system uses Livox Mid360 LiDAR to implement indoor SLAM and mainly consists of the following components:

### 1. LiDAR Driver (`launch/msg_MID360_launch.py`)
- Launches the Livox Mid360 LiDAR
- Publishes point cloud data (`/livox/lidar`) and IMU data (`/livox/imu`)
- Point cloud frequency: 10Hz, field of view: 360°

### 2. SLAM Algorithm (`launch/mapping.launch.py`)
- Subscribes to LiDAR point cloud and IMU data
- Real-time calculation of device position and pose
- Constructs 3D point cloud maps of the surrounding environment
- Publishes localization results (`/Odometry`) and maps (`/cloud_registered`)

### 3. Algorithm Library Source Code (`src/library/`)
- **LiDAR Data Preprocessing** (`preprocess.cpp/h`): Point cloud undistortion, feature extraction
- **IMU Data Processing** (`IMU_Processing.hpp`): Inertial measurement data pre-integration
- **Mapping Algorithm** (`laserMapping.cpp`): Real-time mapping and pose optimization
- **Mathematical Toolkit** (`IKFoM_toolkit/`): Iterative Kalman filter implementation
- **KD-Tree Library** (`ikd-Tree/`): Efficient spatial point cloud management

### 4. Configuration File (`config/mid360.yaml`)
- Sets LiDAR parameters (type, scan lines, etc.)
- Configures IMU parameters (noise covariance, etc.)
- Defines extrinsic parameters (LiDAR position in IMU coordinate system)
- Sets publishing options (path, map, etc.)

## Code Structure Explanation

### Core Algorithm Files:
- **`laserMapping.cpp`**: Main mapping program, implements core SLAM algorithm
- **`preprocess.cpp/h`**: Point cloud preprocessing, extracts feature points
- **`IMU_Processing.hpp`**: IMU data preprocessing and pre-integration

### Mathematical Toolkits:
- **`IKFoM_toolkit/`**: Iterative Extended Kalman Filter toolkit
  - Implements error-state Kalman filtering
  - Handles nonlinear system state estimation
- **`ikd-Tree/`**: Incremental KD-tree library
  - Efficiently manages dynamic point cloud data
  - Supports fast nearest neighbor search

### Auxiliary Tools:
- **`so3_math.h`**: SO(3) group mathematical operations (rotation matrices)
- **`Exp_mat.h`**: Exponential and logarithmic mappings
- **`common_lib.h`**: General mathematical function library

## SLAM Basic Principles

### Data Flow:
```
LiDAR → Point Cloud Data → SLAM Algorithm → Pose Estimation + Map Construction
   ↑                              ↓
IMU Data ←———————————— Sensor Fusion ———————————————
```

### Core Steps:
1. **Sensor Fusion**: Combines LiDAR point cloud and IMU inertial measurement data
2. **Feature Extraction**: Extracts stable geometric features from point clouds (edges, planes)
3. **Pose Estimation**: Calculates device motion based on feature matching
4. **Map Update**: Integrates new point cloud data into the global map

### Key Concepts:
- **Point Cloud**: Collection of 3D spatial points scanned by LiDAR
- **Pose**: Device position and orientation (6 degrees of freedom: x,y,z,roll,pitch,yaw)
- **Odometry**: Relative motion estimation through sensor data
- **Map**: Accumulated geometric representation of the environment

## How to Run

```bash
# 1. Compile the algorithm library (if source code needs modification)
cd src/library
catkin_make  # or colcon build (ROS2)

# 2. Launch LiDAR driver
ros2 launch livox_ros_driver2 msg_MID360_launch.py

# 3. Launch SLAM algorithm (in new terminal)
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml
```

## Observing Results

After launching, you can observe in RViz:
- **Colored Point Cloud**: Real-time scanned LiDAR data
- **Trajectory Path**: Device movement path
- **Global Map**: Accumulated 3D environment map

## Application Scenarios

- Indoor robot navigation
- Building 3D scanning
- Indoor UAV positioning
- Autonomous driving indoor testing

## Important Notes

- Ensure LiDAR is properly connected and powered
- Maintain adequate indoor lighting, avoid direct sunlight
- Move at constant speed, avoid sudden movements
- Regularly save map data to prevent loss
