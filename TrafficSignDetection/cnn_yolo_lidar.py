import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import threading
import math
import time

class DrivingCNN(nn.Module):
    def __init__(self):
        super(DrivingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 9 * 9, 500)
        self.fc2 = nn.Linear(500, 2)  #speed and turn

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 9 * 9)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MoveRobot(Node):
    def __init__(self, model):
        super().__init__('move_robot')
        self.running = True
        self.model = model
        self.class_names = ['not_30', '30', 'right_turn', '2', '1', 'alarm', 'left_turn']

        #yolo - traffic sign
        self.traffic_sign_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/root/yahboomcar_ros2_ws/yahboomcar_ws/yahboom_yolo/yahboom_yolo/best3.pt')
        self.traffic_sign_model.eval()  

        #yolo/deep CNN - vehicle detection 
        self.vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/root/yahboomcar_ros2_ws/yahboomcar_ws/yahboom_yolo/yahboom_yolo/yolo5s.pt')
        self.vehicle_model.eval() 


        #cam
        self.cap = cv2.VideoCapture(0)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.twist = Twist()

        #driving cnn
        self.detection_thread = threading.Thread(target=self.run_camera_detection)
        self.detection_thread.start()

        #lidar for object following
        self.target_distance = 0.23  #this was distance for car
        self.max_speed = 0.07  # Maximum speed of the robot
        self.lidar_subscriber = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)



    def run_camera_detection(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            #yolo
            #['not_30', '30', 'right_turn', '2', '1', 'alarm', 'left_turn']
            results = self.traffic_sign_model(frame)
            for result in results.xyxy[0]:
                class_idx = int(result[5])
                if class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                    print(f"Detected class: {class_name} (index: {class_idx})")
                    #not 30, decrease speed
                    if class_idx == 0:
                        self.move_robot(0.03)
                    #30, increase speed
                    elif class_idx == 1:
                        self.move_robot(0.1)
                    #turn right
                    elif class_idx == 2:
                        self.robot_turn_90()
                    #move forward, turn left into parking space
                    elif class_idx == 3:
                        self.move_robot(self.max_speed)
                        self.robot_turn_90(angular_speed=0.7, clockwise=False)
                    #move forward, turn right into parking space
                    elif class_idx == 4:
                        self.move_robot(self.max_speed)
                        self.robot_turn_90(angular_speed=0.7)
                    #the beeping noise is annoying, so we will print stop and stop robot
                    elif class_idx == 5:
                        self.move_robot.stop()
                        print("stop")
                    elif class_idx == 6:
                        self.robot_turn_90(False)
            

            #driving CNN
            frame_resized = cv2.resize(frame, (100, 100))
            frame_normalized = frame_resized / 255.0
            frame_tensor = torch.tensor(frame_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                action = self.model(frame_tensor).squeeze().numpy()

            
            self.twist.linear.x = float(action[0])  # speed
            self.twist.angular.z = float(action[1]) * 0.75  # turn, used 0.75 for offset
            self.publisher.publish(self.twist)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def lidar_callback(self, msg):
        min_distance = min(msg.ranges)

        if min_distance < self.target_distance:
            #move backwards if too close
            self.move_robot(-self.max_speed)
        elif min_distance > self.target_distance:
            #move forward
            self.move_robot(self.max_speed)
        else:
            self.stop_robot()

    def move_robot(self, speed):
        self.twist.linear.x = speed
        self.twist.angular.z = 0.0
        self.publisher.publish(self.twist)
        print(f"Published command with speed: {speed}")


    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher.publish(self.twist)
        print("Published stop command")

    def robot_turn_90(self, angular_speed=0.5, clockwise=True):
        radians_to_turn = math.pi / 2
        duration_seconds = radians_to_turn / angular_speed
        
        self.twist.linear.x = 0.0  
        self.twist.angular.z = -angular_speed if clockwise else angular_speed  
        
        end_time = self.get_clock().now() + rclpy.time.Duration(seconds=duration_seconds)
        
        while self.get_clock().now() < end_time:
            self.publisher.publish(self.twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.stop_robot_movement()

    def stop_robot_movement(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)

    
    model = DrivingCNN()
    model.load_state_dict(torch.load('/root/yahboomcar_ros2_ws/yahboomcar_ws/yahboom_yolo/yahboom_yolo/best_driving_cnn8.pth'))
    model.eval()  

    move_robot = MoveRobot(model)

    try:
        while rclpy.ok() and move_robot.running:
            rclpy.spin_once(move_robot)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        move_robot.running = False
        move_robot.detection_thread.join()
        move_robot.destroy_node()
        move_robot.stop_robot_movement()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
