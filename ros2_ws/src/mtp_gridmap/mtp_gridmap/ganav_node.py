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

    pallette = np.array([[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ]], dtype=np.uint8)

    def __init__(self):
        super().__init__('ga_nav_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/ga_nav/mask', 10)
        self.subscription_ = self.create_subscription(Image, '/laptop_camera/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()

        # Load ONNX model
        pkg_share = get_package_share_directory('mtp_gridmap')
        model_path = os.path.join(pkg_share, 'models', 'ganav_rugd_6.onnx')
        self.ort_session = ort.InferenceSession(model_path)

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        start_time = time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv_image.astype('float32')[..., ::-1]  # BGR to RGB
        img_resized = cv2.resize(img, (375, 300), interpolation=cv2.INTER_AREA)  # width x height

        # Normalize
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_resized_norm = (img_resized - mean) / std

        # CHW and batch dimension
        img_resized_norm = img_resized_norm.transpose(2, 0, 1)[None, :, :, :]  # 1x3x300x375
        
        onnx_out = self.ort_session.run(None, {'input': img_resized_norm})[0]

        # Create segmentation map
        seg_map = onnx_out.argmax(axis=1)[0]  # Takes maximum
        colored_mask = self.pallette[seg_map]

        overlay = cv2.addWeighted(img_resized.astype(np.uint8), 1 - 0.5, colored_mask, 0.5, 0)
        cv2.imshow("GA-Nav Segmentation Overlay", overlay)
        cv2.waitKey(1)

        # Prepare and publish the mask
        mask_msg = Float32MultiArray()
        mask_msg.data = seg_map.flatten().astype('float32').tolist()
        self.publisher_.publish(mask_msg)
        self.get_logger().info(f'Computed GA-Nav mask in {time() - start_time:.3f} seconds')
        
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