import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointField, PointCloud2
from sensor_msgs_py import point_cloud2
import message_filters
from time import time

class PointCloudNode(Node):
    # Settings and parameters for the Point Cloud Node
    K =  np.array([[690.57118,   0.     , 423.8092],
                   [0.     , 686.79582, 212.43974],
                   [0.     ,   0.     ,   1.     ]])
    
    R_cr = np.array(
        [[0,  0, 1],
        [-1, 0, 0],
        [0, -1, 0]]
    ) # rotation matrix from camera to robot base
    t_cr = np.array([0, 0, 1]) # translation vector from camera to robot base
    n_r = np.array([0, 0, 1]) # plane normal vector in robot base frame
    d_r = 0 # distance from origin to plane along normal in robot base frame
    theta = np.deg2rad(-15) # Angle of camera to the ground - means to the ground as x points to the right of the robot

    c, s = np.cos(theta), np.sin(theta)
    R_angle = np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

    R_cr = R_cr @ R_angle

    def __init__(self):
        super().__init__('point_cloud_node')
        self.bridge = CvBridge()
        self.class_subscription = message_filters.Subscriber(
            self, Image, '/ga_nav/mask'
        )
        self.depth_subscription = message_filters.Subscriber(
            self, Image, '/depth/mask'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.class_subscription, self.depth_subscription],
            queue_size=10,
            slop=0.05 # Is the time difference allowed between messages in seconds
        )
        self.ts.registerCallback(self.sync_callback)

        self.point_cloud_publisher = self.create_publisher(
            PointCloud2, '/point_cloud', 10
        )

        self.get_logger().info('Point Cloud Node has been started.')

    def sync_callback(self, class_msg, depth_msg):
        # Synchronized callback for class and depth images, 'passthrough' encoding keeps original encoding
        class_image = self.bridge.imgmsg_to_cv2(class_msg, desired_encoding='passthrough')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        self.get_logger().info(f'class image shape: {class_image.shape}, depth image shape: {depth_image.shape}')

        start = time()

        point_cloud_msg = self._create_point_cloud(
            class_image, depth_image,
            n_width=100 , n_height=40, 
            header=class_msg.header
        )

        self.point_cloud_publisher.publish(point_cloud_msg)

        self.get_logger().info(f'Point cloud published in {time() - start:.4f} seconds.')


    def _create_point_cloud(self, class_img, depth_img, n_width, n_height, header) -> PointCloud2:
        start_time = time()
        height, width = class_img.shape
        cx = self.K[0,2]
        cy = self.K[1,2]
        fx = self.K[0,0]
        fy = self.K[1,1]

        '''
        The meshgrid creates two ND arrays for each coordinate. The cell it is in represents the respective coordinate of the dimension in a matrix.
        Of the first NDarray does each cell in the meshgrid represents the column coordinate, the second ND array represents the row coordinate.
        Flattening these yields two aligned lists representing all coordinate pairs. Creating a subset grid is possible
        '''
        u, v = np.meshgrid(np.linspace(0, width, n_width, endpoint=False, dtype=np.uint32),
                            np.linspace(0, height, n_height, endpoint=False, dtype=np.uint32))
        u_flat = u.flatten()
        v_flat = v.flatten()
        z_flat = depth_img[v_flat, u_flat].flatten()
        cid_flat = class_img[v_flat, u_flat].flatten()

        x = (u_flat - cx) * z_flat / fx
        y = (v_flat - cy) * z_flat / fy
        z = z_flat
        cid = cid_flat

        # Stack coordinates for coordinate transformation from camera to robot
        points_robot = (self.R_cr @ np.stack((x,y,z))) + self.t_cr.reshape(3,1)
        points_array = np.column_stack((points_robot.T, cid.astype(np.float32)))

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='class_id', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.get_logger().info(f'Min/Max X values: {np.min(points_robot[0,:]):.4f} / {np.max(points_robot[0,:]):.4f}')
        self.get_logger().info(f'Min/Max Y values: {np.min(points_robot[1,:]):.4f} / {np.max(points_robot[1,:]):.4f}')
        self.get_logger().info(f'Min/Max Z values: {np.min(points_robot[2,:]):.4f} / {np.max(points_robot[2,:]):.4f}')
        self.get_logger().info(f'number of points in point cloud: {points_array.shape[0]}')
        header = header
        header.frame_id = 'robot_base'

        return point_cloud2.create_cloud(header, fields, points_array)



def main(args=None):
    rclpy.init(args=args)
    point_cloud_node = PointCloudNode()
    rclpy.spin(point_cloud_node)

    point_cloud_node.destroy_node()
    rclpy.shutdown()
