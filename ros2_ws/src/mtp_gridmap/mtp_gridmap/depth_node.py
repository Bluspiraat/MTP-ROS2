import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import os
import onnxruntime as ort
import cv2
from ament_index_python.packages import get_package_share_directory 
from time import time


# --- DepthNode possible issues ---
#  - The FOV of the input camera might not match the FOV of the camera used during model training. The model was trained with a camera FOV of 60 to 70 degrees.
#  - Depth estimation accuracy may vary based on lighting conditions and scene complexity.


class DepthNode(Node):

    def __init__(self):
        super().__init__('depth_node')
        pkg_share = get_package_share_directory('mtp_gridmap')
        model_weights_path = os.path.join(pkg_share, "models", "depth_vits_392x518.onnx")
        self.publisher_msg_ = self.create_publisher(Image, '/depth/mask', 10)
        self.publisher_msg_color_ = self.create_publisher(Image, '/depth/mask_color', 10)
        self.subscription_ = self.create_subscription(Image, '/image_rect', self.listener_callback, 10)
        self.bridge = CvBridge()

        self.provider_config = {
            'device_id': 0,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': os.path.join(pkg_share, "models", "trt_cache"),
        }   
        os.makedirs(self.provider_config['trt_engine_cache_path'], exist_ok=True)

        self.provider = [('TensorrtExecutionProvider', self.provider_config)] # Either: ['CUDAExecutionProvider'] or [('TensorrtExecutionProvider', self.provider_config)]

        self.onnx_session = ort.InferenceSession(model_weights_path, providers = self.provider)

        self.get_logger().info('Depth Node has been started.')

    def listener_callback(self, msg):
        start_time = time()
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = img_bgr.shape

        # --- Pre-processing --- #
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (518, 392)) # W, H 
        img = img.astype(np.float32) / 255.0 # To float and normalize
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # Normalize to imagenet
        img_input = img.transpose(2, 0, 1)[None] # Transpose to NCHW
        img_input = img_input.astype(np.float32)

        # Infer depth
        start_inference = time()
        depth = self.onnx_session.run(None, {'input': img_input})[0].squeeze()
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