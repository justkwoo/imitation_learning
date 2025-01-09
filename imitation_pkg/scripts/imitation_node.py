#!/usr/bin/env python3

import rclpy
# import pandas as pd
# import matplotlib as plt
import torch
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from imitation_nn import *
from mlp_constants import *


class ImitationLearning(Node):
    def __init__(self):
        super().__init__('imitation_node')
        
        # topics to subscribe or publish
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        
        # create subscriber to Lidar
        self.subsciber = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        
        # create publisher to drive
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
               
        # set up MLP model
        self.model = NeuralNetwork(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, LEARNING_RATE, DEVICE)
        path_to_model = f"/sim_ws/src/imitation_pkg/saved_models/{MODEL_TO_RUN}.pth"
        self.model.load_state_dict(torch.load(path_to_model, weights_only=True))
        
    def lidar_callback(self, data):            
        # create drive msg to publish
        drive_msg = AckermannDriveStamped()
        
        # change dtype to tensor for mlp model
        x_ranges = torch.tensor(data.ranges)
        interval_size = int(1080 / INPUT_LAYER_SIZE)
        x_ranges = x_ranges.view(INPUT_LAYER_SIZE, interval_size).mean(dim=1)

        # pass data to mlp model
        output = self.model.predict_output(x_ranges).tolist()

        # get angle from mlp output
        drive_msg.drive.steering_angle = output[0]
        drive_msg.drive.speed = output[1]
        drive_msg.drive.steering_angle_velocity = 1.5
        
        # publish drive msg 
        self.publisher.publish(drive_msg)
    
    
def main(args=None):
    rclpy.init(args=args)
    print("Imitation Learning Initialized")
    print(f"{MODEL_TO_RUN}")
    print(f"INPUT SIZE: {INPUT_LAYER_SIZE} | HIDDEN SIZE: {HIDDEN_LAYER_SIZE}")
    imitation_node = ImitationLearning()
    
    # destory and shutdown node
    rclpy.spin(imitation_node)
    imitation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()