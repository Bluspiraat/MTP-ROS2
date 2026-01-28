import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, TwistWithCovarianceStamped

class GnssBridge(Node):
    def __init__(self):
        super().__init__('gnss_bridge')
        self.publisher_ = self.create_publisher(TwistWithCovarianceStamped, '/gps/vel_with_cov', 10)
        self.subscription = self.create_subscription(TwistStamped, '/vel', self.listener_callback, 10)

    def listener_callback(self, msg):
        twc = TwistWithCovarianceStamped()
        twc.header = msg.header
        twc.header.frame_id = 'odom'  # Ensure frame is set to 'map'
        twc.twist.twist = msg.twist
        
        # Define covariance (0.05 m/s uncertainty -> 0.0025 variance)
        cov = [0.0] * 36
        cov[0] = 0.5 # x velocity
        cov[7] = 0.5 # y velocity
        twc.twist.covariance = cov
        self.publisher_.publish(twc)

def main(args=None):
    rclpy.init(args=args)
    node = GnssBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()