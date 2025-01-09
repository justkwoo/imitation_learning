#!/usr/bin/env python3

import rclpy
import numpy as np
import math
import torch
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from imitation_nn import *
from mlp_constants import *

class ReactiveFollowGap(Node):
    def __init__(self, model):
        super().__init__('reactive_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # create subscriber to LIDAR
        self.subsciber = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        
        # create publisher to drive
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        
        # disparity variables
        # threshold to identify disparity points
        self.disparity_threshold = 0.5
        
        # constant to reject long ranges
        self.rejectValue = 20.0 
        
        # actual f1tenth car width (not simulator car)
        self.car_width = 0.2032 
        
        # this will guarantee enough space for car to pass the gap
        self.extra_width_factor = 1.9
        
        # speed variables
        self.min_speed = 1.5
        self.max_speed = 6.0
        self.angle_velocity = 1.5 
        self.shift = 3
        self.stretch = 5   
        
        # define models with diff hidden layer size
        self.model = model
        
        # to calculate accuracy
        self.pred_threshold = 0.1
        self.correct_pred = 0 
        self.total_pred = 0
    
    def process_lidar(self, ranges, scan_data):
        # convert ranges to numpy arr
        ranges = np.array(ranges)
        
        # Limit lidar ranges to -60 to 60
        start = int((np.radians(-60) - scan_data.angle_min) / scan_data.angle_increment)
        end = int((np.radians(60) - scan_data.angle_min) / scan_data.angle_increment)
        
        # slice ranges to only include -60 to 60
        new_ranges = ranges[start : end+1] 

        # reject high values
        for i in range(0, len(new_ranges)):
            if new_ranges[i] > self.rejectValue:
                new_ranges[i] = self.rejectValue
        
        return new_ranges
    
    # function to get disparity points
    def get_disparity_points(self, ranges):
        disparities = []
        
        # iterate through the whole range and if higher than threshold append it to the disparity array
        for i in range(1, len(ranges)):
            difference = abs(ranges[i] - ranges[i-1])
            
            # if the difference between two pts is bigger than threshold, append to disparity array
            if (difference > self.disparity_threshold):
               disparities.append(i)
               
        # return array containing dsparity points
        return disparities
    
    def extend_disparity(self, ranges, scan_data):
        # process ranges
        ranges = self.process_lidar(ranges, scan_data)
        
        # get disparity points from processed ranges
        disparities = self.get_disparity_points(ranges)
        
        for i in disparities:
            # get close distance, close pt, and far pt
            close_dist = min(ranges[i], ranges[i-1]) 
            close_index = i if ranges[i] < ranges[i-1] else i-1
            far_index = i if ranges[i] > ranges[i-1] else i-1
            
            # extend the width for safety 
            width = self.car_width * self.extra_width_factor
            
            # get angle 
            angle = 2.0 * np.arcsin(width / (2.0 * close_dist))
            
            # handle NaN case: if angle value is NaN, just go forward (without this it will crash the node)
            if (np.isnan(angle)):
                angle = 0.0
            
            # find number of index to extend
            num_to_extend = int(np.ceil(angle / scan_data.angle_increment)) 
            
            # determine which direction to extend
            isRight = True if close_index < far_index else False
            
            for j in range(num_to_extend):
                # choose index based on direction
                idx = (close_index+j+1) if isRight else (close_index-j-1)
                 
                if idx >= len(ranges):
                    break
                elif idx < 0:
                    continue
                
                if ranges[idx] > close_dist:
                    ranges[idx] = close_dist
            
        return ranges


    def find_steering_angle(self, pt_index, new_ranges_len, angle_increment):
        # compute steering angle to publish and limit it from -90 to 90
        return np.clip((pt_index - (new_ranges_len/2)) * angle_increment, np.radians(-90), np.radians(90))
    
    def determine_speed(self, steering_angle, fp):
        # exponential equation to determine speed (softmax formula from neural network lecture slide)
        speed = (self.max_speed-self.min_speed)/(1+math.pow(self.stretch, -(fp-self.shift))) + self.min_speed
        
        # threshold to decide if it's turning the corner or going on the straight lane
        angle_threshold = np.deg2rad(45)
        
        if abs(steering_angle) > angle_threshold:
            # reduce speed proportionally to the steering angle beyond the threshold
            speed *= max(0.5, 1.0 - (abs(steering_angle) - angle_threshold) / (np.radians(90) - angle_threshold))

        return speed
    
    def compute_accuracy(self):        
        accuracy = round(100*(self.correct_pred/self.total_pred), 3)
        self.get_logger().info(f"MODEL: {INPUT_LAYER_SIZE}_{HIDDEN_LAYER_SIZE} | ACCURACY: {accuracy}%")

    def lidar_callback(self, data):
        # get ranges from scan data
        ranges = data.ranges

        # get extended ranges
        extended_ranges = self.extend_disparity(ranges, data)
    
        # create drive msg to publish
        drive_msg = AckermannDriveStamped()
        
        # determine steering angle
        angle_to_publish = self.find_steering_angle(extended_ranges.argmax(), len(extended_ranges), data.angle_increment)
        drive_msg.drive.steering_angle = angle_to_publish
        
        # return the front range scanned from the extended ranges:
        front_range = extended_ranges[int(len(extended_ranges)/2)]

        # determine speed
        speed_to_publish = self.determine_speed(angle_to_publish, front_range)
        drive_msg.drive.speed = speed_to_publish
        drive_msg.drive.steering_angle_velocity = self.angle_velocity 
        
        # publish msg
        self.publisher.publish(drive_msg)
        
        # change dtype to tensor for mlp model
        ranges_in_tensor = torch.tensor(data.ranges, dtype=torch.float32)
        interval_size = int(1080 / INPUT_LAYER_SIZE)
        ranges_in_tensor = ranges_in_tensor.view(INPUT_LAYER_SIZE, interval_size).mean(dim=1)
        
        # compute loss
        loss = self.model.compute_loss(ranges_in_tensor, angle_to_publish, speed_to_publish)
        self.correct_pred += 1 if loss < self.pred_threshold else 0    
        self.total_pred += 1
    
    
def main(args=None):
    rclpy.init(args=args)
    print("Performance Node Initialized")
    print(f"Testing Model: {MODEL_TO_RUN}")
    
    # set up MLP model
    model = NeuralNetwork(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, LEARNING_RATE, DEVICE)
    
    # load model from disk
    path_to_model = f"/sim_ws/src/imitation_pkg/saved_models/{MODEL_TO_RUN}.pth"
    
    model.load_state_dict(torch.load(path_to_model))
    print("MLP Initialized")
    
    # pass in mlp model
    reactive_node = ReactiveFollowGap(model)
    
    try:
        rclpy.spin(reactive_node)
    except KeyboardInterrupt:
        # ctrl + c to compute accuracy and stop the node
        reactive_node.compute_accuracy()
        reactive_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()