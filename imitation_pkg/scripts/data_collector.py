#!/usr/bin/env python3

import rclpy
import pandas as pd
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from mlp_constants import *

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        
        # subscriber to lidar and drive msg
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, drive_topic, self.drive_callback, 10)
        
        self.current_scan = None
        self.current_drive = None

        self.data = []

    def lidar_callback(self, scan_msg):
        self.current_scan = list(scan_msg.ranges)
    
    def drive_callback(self, drive_msg):
        self.current_drive = drive_msg

        # if we have scan already we just need to combine them 
        if self.current_scan is not None:
            # state-action pair:
            state = self.process_scan(self.current_scan)
            action = self.process_drive(self.current_drive)
            self.data.append([state, action])
            
            # we need to reset it to make sure we have both valid scan and drive data next round
            self.current_scan = None
    
    def process_scan(self, ranges):
        # just in case: can do sth like flatten or average data...
        return ranges
    
    def process_drive(self, drive_msg):
        # extract to [angle, speed]
        # here I only make it float but u can also do some clamping etc.
        angle = float(drive_msg.drive.steering_angle)
        car_speed = float(drive_msg.drive.speed)
        return [angle, car_speed]
    
    def save_data(self):
        # create a dataframe with column state and action
        df = pd.DataFrame(self.data, columns=['state', 'action'])
        df.to_pickle(f'/sim_ws/src/imitation_pkg/expert_data/{DATASET_TO_COLLECT}.pkl')
        self.get_logger().info(f"Saved {len(self.data)} data points")


def main(args=None):
    rclpy.init(args=args)
    print("Data Collector Initialized")
    data_collector_node = DataCollector()
    
    # destory and shutdown node
    # require except to get keyInterrupt so the save_data() func get activated
    try:
        rclpy.spin(data_collector_node)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        data_collector_node.save_data()
        data_collector_node.destroy_node()
        print("Data Collector Done")
        rclpy.shutdown()

if __name__ == '__main__':
    main()