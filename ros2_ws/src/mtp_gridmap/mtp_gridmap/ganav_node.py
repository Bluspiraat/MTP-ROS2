import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import onnxruntime as ort
import numpy as np
import os
import cv2
from ament_index_python.packages import get_package_share_directory 
from time import time

# --- GANavNode possible issues ---
#  - The FOV of the input camera might not match the FOV of the camera used during model training. The model was trained with a camera FOV of 60 to 70 degrees.


class GANavNode(Node):

    pallette = np.array([[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128]], dtype=np.uint8)
    pallette_bgr = pallette[:, ::-1]
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    

    def __init__(self):
        super().__init__('ga_nav_node')
        self.publisher_seg_ = self.create_publisher(Image, '/ga_nav/mask', 10)
        self.publisher_seg_color_ = self.create_publisher(Image, '/ga_nav/mask_color', 10)
        self.subscription_ = self.create_subscription(Image, '/image_rect', self.listener_callback, 10)
        self.bridge = CvBridge()

        # Load ONNX model
        pkg_share = get_package_share_directory('mtp_gridmap')
        model_path = os.path.join(pkg_share, 'models', 'ganav_rugd_6.onnx')

        # Optimization settingss, 2:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(model_path, providers=providers)

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        start_time = time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv_image.astype('float32')[..., ::-1]  # BGR to RGB
        img_resized = cv2.resize(img, (375, 300), interpolation=cv2.INTER_AREA)  # width x height

        # Normalize
        img_resized_norm = (img_resized - self.mean) / self.std

        # CHW and batch dimension
        img_resized_norm = np.ascontiguousarray(img_resized_norm.transpose(2, 0, 1)[None, :, :, :])  # 1x3x300x375
        
        start_inference = time()
        onnx_out = self.ort_session.run(None, {'input': img_resized_norm})[0]
        print(f"Output Shape: {onnx_out.shape}")
        end_inference = time()

        # Create segmentation map
        seg_map = onnx_out.argmax(axis=1)[0]  # Takes maximum

        # Publish to output topic
        '''
        Convert segmentation map to ROS Image message with mono8 encoding
        and publish it. The segmentation values are in range [0, 5] corresponding
        to different classes: 0: background, 1: smooth, 2: rough,
        3: bumpy, 4: forbidden, 5: obstacle.
        '''
        seg_map_int = seg_map.astype(np.uint8)
        seg_msg = self.bridge.cv2_to_imgmsg(seg_map_int, encoding="mono8")
        seg_msg.header = msg.header
        self.publisher_seg_.publish(seg_msg)

        # Color the segmentation map for visualization
        ''' 
        Create a colored overlay for visualization, the classes are mapped as follows:
        0: black (background)
        1: green (smooth)
        2: yellow (rough)
        3: orange (bumpy)
        4: red (forbidden)
        5: dark blue (obstacle)

        It maps the integer segmentation map to a BGR color image using a predefined palette.
        This pallet is defined at the start of the class as `pallette_bgr`.
        '''
        colored_mask = self.pallette_bgr[seg_map]
        color_mask = self.bridge.cv2_to_imgmsg(colored_mask, encoding="bgr8")
        self.publisher_seg_color_.publish(color_mask)
        
        self.get_logger().info(f'Computed GA-Nav mask in {time() - start_time:.3f} seconds with inference in {end_inference - start_inference:.3f} seconds, overhead is {(time() - start_time) - (end_inference - start_inference):.3f} seconds.')
        
def main(args=None):
    rclpy.init(args=args)
    ga_nav_node = GANavNode()
    try:
        rclpy.spin(ga_nav_node)
    finally:
        ga_nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()