import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import os
import torch
import cv2
from ament_index_python.packages import get_package_share_directory 
from time import time
from depth_anything_v2.dpt import DepthAnythingV2


# --- DepthNode possible issues ---
#  - The FOV of the input camera might not match the FOV of the camera used during model training. The model was trained with a camera FOV of 60 to 70 degrees.
#  - Depth estimation accuracy may vary based on lighting conditions and scene complexity.


class DepthNode(Node):
    pkg_share = get_package_share_directory('mtp_gridmap')
    model_weights_path = os.path.join(pkg_share, "models", "depth_anything_v2_metric_vkitti_vits.pth")

    def __init__(self):
        super().__init__('depth_node')
        self.publisher_msg_ = self.create_publisher(Image, '/depth/mask', 10)
        self.publisher_msg_color_ = self.create_publisher(Image, '/depth/mask_color', 10)
        self.subscription_ = self.create_subscription(Image, '/image_rect', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=80)
        self.model.load_state_dict(torch.load(self.model_weights_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    def listener_callback(self, msg):
        start_time = time()
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Infer depth
        start_inference = time()
        depth = self.model.infer_image(img, input_size=518)
        end_inference = time()

        # Publish depth map as ROS Image message
        '''
        Convert disparity map to ROS Image message with 32FC1 encoding and publish it. Add header from the original message.
        '''
        depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = msg.header
        self.publisher_msg_.publish(depth_msg)

        # Publish a visualization of the depth map
        '''
        First, normalize the disparity map to 0-255 and convert to uint8. Then apply a colormap for better visualization.
        Finally, convert to ROS Image message and publish. Header is copied from the original message.
        '''
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
        depth_msg_color = self.bridge.cv2_to_imgmsg(depth_colored, encoding="bgr8")
        depth_msg_color.header = msg.header
        self.publisher_msg_color_.publish(depth_msg_color)

        # Publish computation time information
        self.get_logger().info(f'Computed Depth mask in {time() - start_time:.3f} seconds, with inference time {end_inference - start_inference:.3f} seconds and overhead {(time() - start_time) - (end_inference - start_inference):.3f} seconds.')
        
        
def main(args=None):
    rclpy.init(args=args)
    depth_node = DepthNode()
    try:
        rclpy.spin(depth_node)
    finally:
        depth_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()