import rclpy
import numpy as np
from rclpy.node import Node
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from time import time
import numpy as np

class GridMapNode(Node):
    def __init__(self):
        super().__init__('grid_map_node')
        self.point_cloud_subscription = self.create_subscription(
            PointCloud2,
            '/point_cloud',
            self.listener_callback,
            10
        )
        self.grid_map_publisher = self.create_publisher(
            GridMap,
            '/grid_map',
            10
        )
        self.resolution = 1.0  # meters per cell
        self.x_m = 20.0  # meters
        self.y_m = 20.0  # meters
        self.get_logger().info('Grid Map Node has been started.')

    def listener_callback(self, msg):
        start_time = time()
        point_cloud_np = self.pc_to_numpy(msg)
        cid_map, height_map = self.process_point_cloud(point_cloud_np)
        grid_map_msg = GridMap()
        
        # Adding map information
        grid_map_msg.info.resolution = self.resolution
        grid_map_msg.info.length_x = self.x_m
        grid_map_msg.info.length_y = self.y_m

        # Adding origin information
        grid_map_msg.info.pose.position.x = 10.0
        grid_map_msg.info.pose.position.y = 0.0
        grid_map_msg.info.pose.position.z = 0.0
        
        # Adding pose orientation as a unit quaternion (no rotation)
        grid_map_msg.info.pose.orientation.x = 0.0
        grid_map_msg.info.pose.orientation.y = 0.0
        grid_map_msg.info.pose.orientation.z = -0.5
        grid_map_msg.info.pose.orientation.w = 1.0

        # Adding layers information
        grid_map_msg.layers = ['class_id', 'elevation']

        def create_multi_array(np_array):
            multi_array = Float32MultiArray()
            # GridMap uses a specific storage order. 
            # Flattening should be done carefully.
            flat_data = np_array.astype(np.float32).flatten().tolist()
            multi_array.data = flat_data
            
            # REQUIRED: RViz plugin needs layout dimensions to avoid Segfault
            multi_array.layout.dim.append(MultiArrayDimension(label="column_index", size=np_array.shape[0], stride=np_array.size))
            multi_array.layout.dim.append(MultiArrayDimension(label="row_index", size=np_array.shape[1], stride=np_array.shape[1]))
            return multi_array

        grid_map_msg.data = [
            create_multi_array(cid_map),
            create_multi_array(height_map)
        ]
        # Copying header information
        grid_map_msg.header = msg.header

        # Fill in grid map data based on point cloud processing
        self.grid_map_publisher.publish(grid_map_msg)
        self.get_logger().info(f'Published grid map message in {time() - start_time:.3f} seconds.')

    def process_point_cloud(self, point_cloud):
        # Convert point cloud to grid map representation
        y_count, x_count = int(self.y_m / self.resolution), int(self.x_m / self.resolution)
        y_offset = int(self.y_m / (2 * self.resolution)) # Shift values to positive range

        # Calculate the correct bin for each point
        xs = np.floor(point_cloud[:,0] / self.resolution).astype(np.int32)
        ys = np.floor(point_cloud[:,1] / self.resolution).astype(np.int32) + y_offset
        zs = point_cloud[:,2]
        cids = point_cloud[:,3].astype(np.int32)

        # Filter points within grid map bounds
        mask = (xs >= 0) & (xs < x_count) & (ys >= 0) & (ys < y_count)
        xs, ys, zs, cids = xs[mask], ys[mask], zs[mask], cids[mask]

        # Describe each bin as a unique index
        bin_indices = ys * x_count + xs

        # Get the index order such that all indices are grouped per cell and apply this order to the height and cids.
        sort_idx = np.argsort(bin_indices)
        bin_indices = bin_indices[sort_idx]
        zs = zs[sort_idx]
        cids = cids[sort_idx]

        # Find where the cell changes and create 'slices' [0 .... changes ... last index]
        diffs = np.concatenate(([0], np.flatnonzero(np.diff(bin_indices)) + 1, [len(bin_indices)]))
        
        # Create output grids
        h_grid = np.full((x_count, y_count), np.nan)
        cid_grid = np.zeros((x_count, y_count), dtype=np.int32)

        # Go over each bin, get the unique cell value, cell ids and the height values.
        for i in range(len(diffs) - 1):
            idx_start, idx_end = diffs[i], diffs[i + 1]
            cell_idx = bin_indices[idx_start]
            y_idx = cell_idx // x_count
            x_idx = cell_idx % x_count
            h_grid[x_idx, y_idx] = np.mean(zs[idx_start:idx_end])

            # Mode is still the slowest part
            cid_grid[x_idx, y_idx] = np.argmax(np.bincount(cids[idx_start:idx_end]))
        
        return np.rot90(np.fliplr(cid_grid), 3), np.rot90(np.fliplr(h_grid), 3)

    def pc_to_numpy(self, point_cloud_msg):
        # Convert PointCloud2 message to numpy array
        pc_data = point_cloud2.read_points(point_cloud_msg, field_names=['x', 'y', 'z', 'class_id'], skip_nans=True)
    
        # Convert generator to a standard numpy matrix, view() is used to interpret the structured data as a flat float32 array.
        # Each point is loaded into a numpy array and each point is a numpy.void class which has to be converted to float32.
        points = np.array(list(pc_data)).view(np.float32).reshape(-1, 4)
        return points

def main(args=None):
    rclpy.init(args=args)
    grid_map_node = GridMapNode()
    rclpy.spin(grid_map_node)

    grid_map_node.destroy_node()
    rclpy.shutdown()
