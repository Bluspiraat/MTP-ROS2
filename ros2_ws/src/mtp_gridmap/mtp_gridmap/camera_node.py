import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class LaptopCameraNode(Node):
    def __init__(self):
        super().__init__('laptop_camera_node')
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, '/laptop_camera/image_raw', 10)
        self.cap = cv2.VideoCapture(0)

        self.timer = self.create_timer(0.2, self.timer_callback)  # 5 Hz

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
            cv2.imshow('Laptop Camera', frame)
            cv2.waitKey(1)
        else:
            self.get_logger().error('Failed to capture image from camera')

def main(args=None):
    rclpy.init(args=args)
    laptop_camera_node = LaptopCameraNode()
    rclpy.spin(laptop_camera_node)

    laptop_camera_node.cap.release()
    laptop_camera_node.destroy_node()
    rclpy.shutdown()