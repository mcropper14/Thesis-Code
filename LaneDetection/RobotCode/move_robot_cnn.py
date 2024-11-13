import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            cv2.imshow('Video Feed', frame) #can turn on or off 
            frame_resized = cv2.resize(frame, (100, 100))
            frame_normalized = frame_resized / 255.0
            frame_tensor = torch.tensor(frame_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor and add batch dimension
            return frame_tensor
        else:
            return None

    def step(self, action):
        twist = Twist()
        twist.linear.x = float(action[0])  #speed
        twist.angular.z = float(action[1])  #turn
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

    # Load the PyTorch model
    model = DrivingCNN()
    model.load_state_dict(torch.load('best_driving_cnn.pth'))
    model.eval()  

    env = RLEnvironment(model)

    try:
        while rclpy.ok():
            state = env.get_state()
            if state is not None:
                with torch.no_grad():
                    action = model(state).squeeze().numpy()  #predict speed and turn
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
