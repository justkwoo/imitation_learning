#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import pandas as pd
from datetime import datetime
import sys
import select
import tty
import termios
import numpy as np
from time import sleep
from mlp_constants import *

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        # crf topic is used for conflict resolution
        crf_drive_topic = '/crf_drive_topic'
        
        # subscriber to lidar and drive msg
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        # self.drive_sub = self.create_subscription(AckermannDriveStamped, drive_topic, self.drive_callback, 10)
        # for crf purposes
        self.drive_sub = self.create_subscription(AckermannDriveStamped, drive_topic, self.drive_callback, 10)
        
        self.current_scan = None
        self.current_drive = None

        self.expertData = None

        self.reCollectDelay = 2 # 2 sec delay after crashed or getting too close

        self.expertNum = 2

        self.data = []

    def lidar_callback(self, scan_msg):
        self.current_scan = list(scan_msg.ranges)

    def drive_callback(self, drive_msg):
        self.current_drive = drive_msg

        # anaylze which expert:
        self.current_drive.drive.steering_angle_velocity

        # if we have scan already we just need to combine them 
        if self.current_scan is not None:
            # state-action pair:
            state = self.process_scan(self.current_scan)
            action = self.process_drive(self.current_drive)
            if self.CBFdatafilter(self.current_scan):
                data_to_remove = 20
                for _ in range(data_to_remove):
                    temp = self.data.pop()
            else:
                self.data.append([state, action])
                # we need to reset it to make sure we have both valid scan and drive data next round
                self.current_scan = None

    def CBFdatafilter(self, scan_msg):
        # require current ego position
        # require current obstacle position
        safety_distance = 0.3
        min_dist = np.min(np.array(scan_msg))
        self.get_logger().info(f"{min_dist}")
        if (min_dist <= safety_distance):
            sleep(self.reCollectDelay)
            self.get_logger().info(f"triggered removing points")
            return True
        return False

    def datafilter(self):
        data_to_remove = 20
        before_len = len(self.data)
        for _ in range(data_to_remove):
            temp = self.data.pop()
        after_len = len(self.data)
        self.get_logger().info(f"removed {data_to_remove} data points collected (from {before_len} to {after_len})")
    
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
        # just making the name unique
        current_date = datetime.now().strftime('%m-%d_%H-%M')
        df.to_pickle(f'/sim_ws/src/imitation_pkg/expert_data/{DATASET_TO_COLLECT}.pkl')
        self.get_logger().info(f"GG Saved {len(self.data)} data points :P")

def main(args=None):
    rclpy.init(args=args)
    print("Data Collector Initialized")
    data_collector_node = DataCollector()

    # fd = sys.stdin.fileno()
    # old_settings = termios.tcgetattr(fd)
    
    # destory and shutdown node
    # require except to get keyInterrupt so the save_data() func get activated
    try:
        # rclpy.spin(data_collector_node)
        # tty.setcbreak(fd)
        rclpy.spin(data_collector_node)
            # dr, dw, de = select.select([sys.stdin], [], [], 0)
            # if dr:
            #     c = sys.stdin.read(1)
            #     if c.lower() == 'c':
            #         data_collector_node.datafilter()
    except KeyboardInterrupt:
        data_collector_node.save_data()
        data_collector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()