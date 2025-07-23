# Isaac Sim Procedural Terrain & Sensor Simulation Framework

> ⚙️ In development as part of an internship project at **Georgia Tech Europe – DREAM LABS**

This repository contains a modular simulation framework built in **NVIDIA Isaac Sim**, focused on large-scale procedural terrain generation and **real-time ROS2 sensor integration** for robotics research and testing.

## 🧠 Purpose
The goal of this project is to create a **realistic simulation environment** for autonomous robots to:
- Navigate dynamically generated terrains
- Perceive the world using ROS2-compatible sensors (LiDAR, IMU, cameras)
- Visualize sensor data in **RViz**
- Extract and record meaningful data streams (e.g. `rosbag`) for **predictive perception** and **foundation model training**

## 🛠️ Features (In Progress)
- [x] Real-time heightmap-based terrain generation using Perlin noise  
- [x] Dynamic terrain tile streaming as the robot moves  
- [x] Poisson-disk object instancing (trees, grass) with semantic scaling  
- [x] ROS2 sensor publishing (LiDAR, IMU, RGB camera)  
- [x] Full RViz visualization (point clouds, transforms, camera feeds)  
- [ ] Terrain-aware navigation and map-saving  
- [ ] Integration with visuo-tactile perception pipelines  

## 🔧 Technologies
- **Isaac Sim 4.5.0**
- **ROS2 Humble**
- **Python 3.10**
- `omni.replicator`, `omni.isaac.core`, `ros2_bridge`, `NumPy`, `OpenCV`

## 📍 Status
This project is actively in development and will continue to evolve over the course of the internship.

---

Feel free to fork, follow, or reach out for collaboration!

