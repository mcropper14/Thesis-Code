# Thesis-Code

# Multi-Sensor Fusion and V2V Communication in Autonomous Vehicles: A Comprehensive Framework for Self-Driving

Welcome to the code repository accompanying the paper *Multi-Sensor Fusion and V2V Communication in Autonomous Vehicles: A Comprehensive Framework for Self-Driving*. This repository provides the code and configuration necessary to replicate and extend the experiments, simulations, and system setup described in the paper. It includes various folders that directly correspond to sections within the research paper, each representing distinct components of our proposed autonomous vehicle framework.

## Repository Structure

This repository is organized into several key folders, each of which holds the code and resources relevant to specific aspects of the autonomous vehicle system. Below is a summary of each folder and its purpose:

- **3D Visualization**: Contains scripts and tools for 3D visualization of the vehicle's environment, allowing for real-time rendering and analysis of the multi-sensor data collected by the autonomous system.
  
- **LaneDetection**: Implements the lane detection algorithms discussed in the paper, enabling the vehicle to identify and track lane boundaries in various driving conditions.
  
- **Setup**: Contains the setup scripts, configurations, and Docker container necessary for initializing the development environment. Please refer to this folder for instructions on environment setup.

- **Supercombo**: This contains the code used to run Comma AI's Supercombo model and display the visualization that was presented in the paper. 
  
- **TrafficSignDetection**: Contains the implementation of traffic sign recognition algorithms, enabling the vehicle to detect and classify traffic signs.
  
- **Vehicle-to-Vehicle**: Implements the vehicle-to-vehicle (V2V) communication protocols, allowing for data exchange between autonomous vehicles to improve situational awareness.
  
- **VehicleDetection**: Contains the code for vehicle detection, using sensor input to identify and track other vehicles in the environment.

Each folder contains scripts, data, and models relevant to the corresponding section in the paper.

## Setup Instructions

The **Setup** folder includes a Docker container with all required dependencies and scripts for running the system. To initialize the Docker environment, navigate to the Setup folder and execute the following command:

```bash
./run_docker.sh

