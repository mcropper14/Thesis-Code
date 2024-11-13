"""
Loads in the model as an onnx to make use of TensorRT for faster inference.
Save as move_robot_tensorRT.py, just uses tensorRT to speed up inference.
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tensorrt_model import TensorRTModel

class RLEnvironment(Node):
    def __init__(self, model):
        super().__init__('rl_environment')
        self.model = model  
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.cap = cv2.VideoCapture(0)
        self.state = None

    def get_state(self):
        ret, frame = self.cap.read()
        if ret:
            #cv2.imshow('Video Feed', frame) 
            frame_resized = cv2.resize(frame, (64, 64))  
            frame_normalized = frame_resized / 255.0
            frame_np = np.transpose(frame_normalized, (2, 0, 1)).astype(np.float32)  
            frame_np = np.expand_dims(frame_np, axis=0)  
            return frame_np  
        else:
            return None

    def step(self, action):
        twist = Twist()
        twist.linear.x = float(action[0])   
        twist.angular.z = float(action[1]) 
        self.publisher.publish(twist)

        rclpy.spin_once(self, timeout_sec=0.1)

        next_state = self.get_state()
        done = self.is_done()

        return next_state, done

    def is_done(self):
        return False

    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)

    onnx_path = 'best_driving_cnn8.onnx'  
    trt_path = 'best_driving_cnn8.trt'    
    model = TensorRTModel(onnx_path, trt_path)
    env = RLEnvironment(model)
    try:
        while rclpy.ok():
            state = env.get_state()
            if state is not None:
                action = model.predict(state).squeeze()  
                next_state, done = env.step(action)
                if done:
                    break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.stop()
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
