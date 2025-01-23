# Thesis-Code

# Multi-Sensor Fusion and V2V Communication in Autonomous Vehicles: A Comprehensive Framework for Self-Driving

Welcome to the code repository accompanying the paper *Multi-Sensor Fusion and V2V Communication in Autonomous Vehicles: A Comprehensive Framework for Self-Driving*. This repository provides the code and configuration necessary to replicate results, figures, and system setup described in the paper. It includes various folders that directly correspond to sections within the research paper, each representing distinct components of our proposed autonomous vehicle framework.

## Paper

Download the paper from here: [https://scholarworks.wm.edu/honorstheses/2243/]

## Demo Videos

Demo: [https://youtu.be/pAVnPyNsMko]
Presentation: [https://youtu.be/s9K2VZNC5XY]

## System Overview

The robot scripts in this repository are designed to run in an Ubuntu 20.04 container on a ROSMASTER X3 robot. This configuration uses the ROS2 (Robot Operating System) middleware to handle communication between different components such as movement control, LiDAR sensing, and vehicle-to-vehicle communication.

## Repository Structure

This repository is organized into several key folders, each of which holds the code and resources relevant to specific aspects of the autonomous vehicle system. Below is a summary of each folder, along with descriptions of the scripts within them:

- **LaneDetection**: Implements the lane detection algorithms discussed in the paper.
  - `CNN/`: Contains scripts for Convolutional Neural Network models used in lane detection.
    - `cnn_no_lstm.py`: CNN model without LSTM for lane detection.
    - `efficient.py`: EfficientNet-based model with LSTM implementation for lane detection.
    - `feature_map.py`: Generates feature maps for visualizing lane boundaries.
    - `heat_map.py`: Creates heat maps for lane region visualization.
  - `RobotCode/`: Includes code for integrating lane detection with robot control. These are scripts that must be ran on the robot.
    - `cnn_yolo_lidar.py`: Combines CNN-based lane detection with LiDAR data for precise localization.
    - `data_collect_speed.py`: Collects data related to vehicle speed for training the CNN.
    - `move_robot_cnn.py`: Script for controlling robot movement based on CNN model predictions.
    - `move_robot_tensorRT.py`: Robot movement control using TensorRT-optimized model.
    - `TensorRT.py`: TensorRT model optimization script for faster inference.

- **Setup**: Contains the setup scripts, configurations, and Docker container necessary for initializing the development environment.
  - `run_docker.sh`: Script to build and start the Docker container.
  - `start_nodes.txt`: Instructions for starting the necessary ROS2 nodes.

- **Supercombo**: This contains the code used to run Comma AI's Supercombo model and display the visualization that was presented in the paper. 


- **TrafficSignDetection**: Contains the implementation of traffic sign recognition algorithms.
  - `augment.py`: Data augmentation script for traffic sign images.
  - `cnn_yolo_lidar.py`: Combines CNN-based traffic sign detection with LiDAR data.
  - `organize_data.py`: Organizes and preprocesses traffic sign data for model training.

- **Vehicle-to-Vehicle**: Implements the vehicle-to-vehicle (V2V) communication protocols.
  - `Robot_1.py`: Script for V2V communication and control for vehicle 1.
  - `Robot_2.py`: Script for V2V communication and control for vehicle 2.

- **VehicleDetection**: Contains the code for vehicle detection, using sensor input to identify and track other vehicles.
  - `detect_lidar_follow.py`: Detects vehicles using LiDAR data and enables the following behavior.
  - `detect.py`: Vehicle detection script based on image and sensor data.
  - `lidar_follow.py`: Implements following behavior based on LiDAR-detected vehicles.

Each folder contains scripts, data, and models relevant to the corresponding section in the paper, allowing users to explore and implement each component individually.

## Setup Instructions

The **Setup** folder includes a Docker container with all required dependencies and scripts for running the system. To initialize the Docker environment, navigate to the Setup folder and execute the following command:

```bash
./run_docker.sh

