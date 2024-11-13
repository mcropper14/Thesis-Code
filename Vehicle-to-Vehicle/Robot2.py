"""
Run on Robot 2 ("client")
Each Robot needs to have the same Domain ID

"""


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
import math
from rclpy.duration import Duration

class MoveRobot2(Node):
    def __init__(self):
        super().__init__('move_robot_2')

        self.publisher = self.create_publisher(Twist, 'cmd_vel_2', 10)
        self.pose_publisher = self.create_publisher(Pose, 'robot2_pose', 10)
        self.pose_subscriber = self.create_subscription(Pose, 'robot1_pose', self.pose_callback, 10)
        
        self.twist = Twist()
        self.other_robot_pose = Pose()

    def pose_callback(self, msg):
        self.other_robot_pose = msg  

    def broadcast_pose(self, position, orientation):
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.orientation.z = orientation

        self.pose_publisher.publish(pose)

    def move_robot_forward(self, duration_seconds, speed=0.07):
        self.twist.linear.x = speed
        self.twist.angular.z = 0.0

        end_time = self.get_clock().now() + Duration(seconds=duration_seconds)

        while self.get_clock().now() < end_time:
            self.publisher.publish(self.twist)
            self.broadcast_pose([self.twist.linear.x, 0.0], self.twist.angular.z)  # Broadcast position
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.detect_potential_collision():
                self.stop_robot_movement()
                break

        self.stop_robot_movement()

    def robot_turn_90(self, angular_speed=0.5, clockwise=True):
        radians_to_turn = math.pi / 2
        duration_seconds = radians_to_turn / angular_speed

        self.twist.linear.x = 0.0
        self.twist.angular.z = -angular_speed if clockwise else angular_speed

        end_time = self.get_clock().now() + Duration(seconds=duration_seconds)

        while self.get_clock().now() < end_time:
            self.publisher.publish(self.twist)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.stop_robot_movement()

    def detect_potential_collision(self):
        #just the distance formula
        distance_between_robots = math.sqrt((self.twist.linear.x - self.other_robot_pose.position.x)**2 +
                                            (0.0 - self.other_robot_pose.position.y)**2)
        return distance_between_robots < 0.5  #0.5 meters 

    def stop_robot_movement(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    move_robot_2 = MoveRobot2()
    
    try:
        move_robot_2.move_robot_forward(5) 
    except KeyboardInterrupt:
        pass
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
