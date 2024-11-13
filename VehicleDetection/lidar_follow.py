import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class FollowObject(Node):
    def __init__(self):
        super().__init__('follow_object')

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.twist = Twist()
        self.lidar_subscriber = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        self.target_distance = 0.23  
        self.max_speed = 0.07  
    
    def lidar_callback(self, msg):
        min_distance = min(msg.ranges)
        print(f"Min Distance: {min_distance}")

        if min_distance < self.target_distance:
            self.move_robot(-self.max_speed)
        elif min_distance > self.target_distance:
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

def main(args=None):
    rclpy.init(args=args)
    follow_object = FollowObject()
    rclpy.spin(follow_object)
    follow_object.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()