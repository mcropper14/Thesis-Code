
"""
Robot Data Collection Node
This node collects speed and turn data from the robot and saves it to a CSV file.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import os

class SpeedTurnDataCollector(Node):
   def __init__(self):
       super().__init__('speed_turn_data_collector')
       self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
       self.output_folder = 'collected_speed_turn_data'
       self.csv_file = os.path.join(self.output_folder, 'speed_turn_data.csv')

       if not os.path.exists(self.output_folder):
           os.makedirs(self.output_folder)
       
       #CSV file for speed and turn data
       with open(self.csv_file, 'w', newline='') as csvfile:
           csvwriter = csv.writer(csvfile)
           csvwriter.writerow(['frame_count', 'speed', 'turn'])

       self.frame_count = 0
       self.speed = None
       self.turn = None

       self.get_logger().info('SpeedTurnDataCollector node has been initialized.')

   def cmd_vel_callback(self, msg):
       # Extract speed and turn data
       self.speed = msg.linear.x
       self.turn = msg.angular.z
       self.frame_count += 1
       
       #save data every x frames, 5 or lower is good 
       if self.frame_count % 5 == 0:
           self.save_speed_turn_data()
           self.get_logger().info(f'Collected data for frame {self.frame_count}: speed={self.speed}, turn={self.turn}')

   def save_speed_turn_data(self):
       with open(self.csv_file, 'a', newline='') as csvfile:
           csvwriter = csv.writer(csvfile)
           csvwriter.writerow([self.frame_count, self.speed, self.turn])
       self.get_logger().info(f'Saved speed and turn data for frame {self.frame_count}')

def main():
   rclpy.init()
   speed_turn_data_collector = SpeedTurnDataCollector()
   try:
       rclpy.spin(speed_turn_data_collector)
   except KeyboardInterrupt:
       pass
   finally:
       speed_turn_data_collector.destroy_node()
       rclpy.shutdown()

if __name__ == '__main__':
   main()
