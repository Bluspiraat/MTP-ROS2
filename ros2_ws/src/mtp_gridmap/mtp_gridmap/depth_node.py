import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import os
import torch
from torchvision import transforms
import cv2
from mono_depth.layers import disp_to_depth
from mono_depth import networks
from ament_index_python.packages import get_package_share_directory 
from time import time

# --- DepthNode possible issues ---
#  - The FOV of the input camera might not match the FOV of the camera used during model training. The model was trained with a camera FOV of 60 to 70 degrees.
#  - Depth estimation accuracy may vary based on lighting conditions and scene complexity.


class DepthNode(Node):
    pkg_share = get_package_share_directory('mtp_gridmap')
    encoder_path = os.path.join(pkg_share, "models", "lm_8m_encoder.pth")
    decoder_path = os.path.join(pkg_share, "models", "lm_8m_depth.pth")
    model_name = "lite-mono-8m"

    def __init__(self):
        super().__init__('depth_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/depth/mask', 10)
        self.subscription_ = self.create_subscription(Image, '/laptop_camera/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder, self.decoder, self.feed_height, self.feed_width = self.load_model()

    def load_model(self):
        encoder_dict = torch.load(self.encoder_path, map_location=self.device)
        decoder_dict = torch.load(self.decoder_path, map_location=self.device)

        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']

        encoder = networks.LiteMono(model=self.model_name, height=feed_height, width=feed_width)
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        encoder.to(self.device).eval()

        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})
        depth_decoder.to(self.device).eval()

        return encoder, depth_decoder, feed_height, feed_width
            

    def listener_callback(self, msg):
        start_time = time()
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        original_height, original_width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.feed_width, self.feed_height))
        input_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            start_inference = time()
            features = self.encoder(input_tensor)
            outputs = self.decoder(features)
            end_inference = time()
            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

            disp_resized_np = disp_resized.squeeze().cpu().numpy()

            # Assume disp_resized_np is a 2D NumPy array
            disp = disp_resized_np.copy()

            # Clip values to avoid extreme outliers (optional, like vmax in matplotlib)
            vmax = np.percentile(disp, 95)
            disp = np.clip(disp, 0, vmax)

            # Normalize to 0-255
            disp_normalized = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)

            # Apply OpenCV colormap (COLORMAP_MAGMA is similar to matplotlib 'magma')
            disp_colored = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_MAGMA)

            # Show with OpenCV
            cv2.imshow("Depth Heatmap", disp_colored)
            cv2.waitKey(1)
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