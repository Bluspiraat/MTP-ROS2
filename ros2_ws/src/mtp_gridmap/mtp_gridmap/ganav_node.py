import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import onnxruntime as ort
import numpy as np
import os
import cv2

class GANavNode(Node):

    pallette = np.array([[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ]], dtype=np.uint8)

    def __init__(self):
        super().__init__('ga_nav_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/ga_nav/mask', 10)
        self.subscription_ = self.create_subscription(Image, '/laptop_camera/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()

        # Load ONNX model
        self.ort_session = ort.InferenceSession('/ros2_ws/src/mtp_gridmap/models/ganav_rugd_6.onnx')

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
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