import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Quaternion
import math

class GpsHeadingNode(Node):
    def __init__(self):
        super().__init__('gps_heading_node')
        
        # Subscriptions and Publishers
        self.subscription = self.create_subscription(NavSatFix, '/fix', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Imu, '/heading', 10)
        
        self.prev_fix = None
        # Threshold: Only calculate heading if robot moved > 0.5 meters to avoid noise jitter
        self.distance_threshold = 0.5 

    def listener_callback(self, current_fix):
        if self.prev_fix is None:
            self.prev_fix = current_fix
            return

        # 1. Calculate distance between points (approximate for short distances)
        d_lat = (current_fix.latitude - self.prev_fix.latitude) * 111320
        d_lon = (current_fix.longitude - self.prev_fix.longitude) * (111320 * math.cos(math.radians(current_fix.latitude)))
        distance = math.sqrt(d_lat**2 + d_lon**2)

        if distance > self.distance_threshold:
            # 2. Calculate Bearing (atan2(delta_lon, delta_lat))
            # In standard math, atan2(y, x) where y is East/West and x is North/South
            # This gives 0 = North, clockwise positive.
            bearing = math.atan2(d_lon, d_lat)

            # 3. Convert to ROS ENU (East-North-Up)
            # ROS Yaw: 0 = East (+X), counter-clockwise positive
            ros_yaw = -(bearing - (math.pi / 2))

            # 4. Populate the IMU Message
            imu_msg = Imu()
            imu_msg.header = current_fix.header
            imu_msg.header.frame_id = "gnss_link" # Or your IMU frame
            
            # Convert Yaw to Quaternion
            imu_msg.orientation = self.get_quaternion_from_euler(0, 0, ros_yaw)

            # 5. Assign Covariance (Crucial for EKF!)
            # Since this is a derived value, give it a medium-high variance
            # notes that 0.0 covariance causes EKF issues.
            # 0.04 variance is approx 0.2 rad (11 degrees) of uncertainty.
            imu_msg.orientation_covariance[8] = 0.04 

            self.publisher_.publish(imu_msg)
            self.prev_fix = current_fix

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

def main(args=None):
    rclpy.init(args=args)
    node = GpsHeadingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.init()

if __name__ == '__main__':
    main()