import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import torch
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class FollowObject(Node):
    def __init__(self):
        super().__init__('follow_object')

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.lidar_subscriber = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        self.camera_subscriber = self.create_subscription(Image, 'camera/image_raw', self.camera_callback, 10)
        self.twist = Twist()
        self.target_distance = 0.23  
        self.max_speed = 0.07  

        #get the vehicle model 
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  
        self.model.eval()  

        self.bridge = CvBridge()
        self.vehicle_detected = False

    def lidar_callback(self, msg):
        min_distance = min(msg.ranges)
        print(f"Min Distance: {min_distance}")

        if self.vehicle_detected:
            if min_distance < self.target_distance:
                self.move_robot(-self.max_speed)
            elif min_distance > self.target_distance:
                self.move_robot(self.max_speed)
            else:
                self.stop_robot()

    def camera_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        results = self.model(frame)
        detections = results.pandas().xyxy[0] 

        self.vehicle_detected = False
        for _, detection in detections.iterrows():
            if detection['class'] == 2 and detection['confidence'] > 0.50:
                self.vehicle_detected = True
                break 

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

def main(args=None):
    rclpy.init(args=args)
    follow_object = FollowObject()
    rclpy.spin(follow_object)
    follow_object.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
