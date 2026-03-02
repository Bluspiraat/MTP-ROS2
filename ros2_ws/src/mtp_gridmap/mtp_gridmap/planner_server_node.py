import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionServer
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker
from mtp_planner_interfaces.action import NavigateToGPS  # Import custom action
from rclpy.callback_groups import ReentrantCallbackGroup
from .planner_algorithm import AstarROS

def calculate_heading_and_distance(curr_lat, curr_long, target_lat, target_long):
    # Haversine formula or simple Euclidean approximation for short distances
    d_lat = (target_lat - curr_lat) * 111320  # Meters per degree latitude measured from the equator
    d_lon = (target_long - curr_long) * (111320 * np.cos(np.radians(target_lat)))  # Adjust for longitude convergence --> Longitude lines converge towards poles
    distance = np.sqrt(d_lat**2 + d_lon**2)
    return distance, np.arctan2(d_lat, d_lon)  # Return distance and bearing

def bring_goal_in_bounds(x_m, y_m, angle, distance):
    # Angle from origin to the corner of the map to determine which side to intersect with
    angle_to_corner = np.arctan2(y_m/2, x_m)
    if abs(angle) < angle_to_corner:
        # Intersecting front
        return x_m, np.tan(angle) * x_m
    elif y_m > 0 and angle > angle_to_corner:
        # Front left quadrant intersecting side
        return np.tan(np.pi/2 - angle) * y_m/2, y_m/2
    else:  # y_m < 0 and angle > -angle_to_corner
        # Front right quadrant intersecting side
        return np.tan(np.pi/2 - abs(angle)) * y_m/2, -y_m/2

def calculate_position(angle, distance):
    # Calculate position in 
    goal_pos_x = distance * np.cos(angle)
    goal_pos_y = distance * np.sin(angle)
    return goal_pos_x, goal_pos_y

def grid_map_to_numpy(grid_map_msg, layer_name, fill_nan=False, nan_value=None):
    layer_index = grid_map_msg.layers.index(layer_name)
    row_count = int(grid_map_msg.info.length_x / grid_map_msg.info.resolution)
    column_count = int(grid_map_msg.info.length_y / grid_map_msg.info.resolution)

    grid = np.array(grid_map_msg.data[layer_index].data).reshape((row_count, column_count))
    if fill_nan:
        np.nan_to_num(grid, copy=False, nan=nan_value)  # Replace NaN values with nan_value (indicating non-traversable areas)

    start_index_x = grid_map_msg.outer_start_index
    start_index_y = grid_map_msg.inner_start_index

    # Read through why this is needed --> Something with moving the robot and circular buffers.
    if start_index_x != 0 or start_index_y != 0:
        grid = np.roll(grid, shift=(-start_index_x, -start_index_y), axis=(0, 1))

    return grid

def calculate_path(start_cell, goal_cell, height_grid, cid_grid, cost_list, resolution):
    """
    The segmentation values are in range [0, 5] corresponding to different classes:
    0: background, 1: smooth, 2: rough, 3: bumpy, 4: forbidden, 5: obstacle.
    """
    traverse_cost = np.array(cost_list)[cid_grid.astype(int)]  # Map class_id to traversability cost using the cost_list
    planner = AstarROS(height_grid, traverse_cost, resolution)
    path_list = planner.start_search(start_cell, goal_cell)
    return path_list

class PlannerServer(Node):
    def __init__(self):
        super().__init__('planner_server_node')

        self.current_latitude = 52.2399
        self.current_longitude = 6.8398
        self.current_yaw = 0.0
        self.target_latitude = None
        self.target_longitude = None
        self.threshold = None
        self.grid_map = None

        self.group = ReentrantCallbackGroup()  # For handling multiple callbacks concurrently if needed
        
        # Subscribers
        self.sub_yaw = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.yaw_callback,
            10,
            callback_group=self.group
        )

        self.sub_pos = self.create_subscription(
            NavSatFix,
            '/gps/filtered',
            self.gps_callback,
            10,
            callback_group=self.group
        )
        
        self.sub_costmap = self.create_subscription(
            GridMap,
            '/grid_map',
            self.cost_map_callback,
            10,
            callback_group=self.group
        )

        self.pub_path = self.create_publisher(
            Path, 
            '/planned_path', 
            10)  # Placeholder for path publisher, can be changed to a custom message type for paths

        self.pub_goal = self.create_publisher(
            Marker,
            '/goal_point',
            10
        )

        self.pub_start = self.create_publisher(
            Marker,
            '/start_point',
            10
        )

        # Action Server
        self._action_server = ActionServer(
            self, NavigateToGPS, 'navigate_gps', 
            execute_callback=self.execute_callback,
            callback_group=self.group
        )  
        
        self.get_logger().info('Planner Server Node has been started.')

    def yaw_callback(self, msg):
        # Process odometry data --> calculate current yaw of robot wrt ENU.
        q = msg.pose.pose.orientation
        self.current_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        # self.get_logger().info(f'Current yaw: {self.current_yaw}')

    def gps_callback(self, msg):
        # Process GPS data
        self.current_latitude = msg.latitude
        self.current_longitude = msg.longitude
        # self.get_logger().info(f'Received GPS data: {self.current_latitude}, {self.current_longitude}')
    
    def cost_map_callback(self, msg):
        # Process cost map data
        self.grid_map = msg
        # self.get_logger().info(f'Received cost map with layers: {msg.layers} at timestamp {msg.header.stamp.sec}')

    def goal_callback(self, goal_request):
        target_latitude = goal_request.request.latitude
        target_longitude = goal_request.request.longitude
        if target_latitude > 90.0 or target_latitude < -90.0 or target_longitude > 180.0 or target_longitude < -180.0:
            return GoalResponse.REJECT
        GoalResponse.ACCEPT

    def publish_marker(self, x, y, marker_id, color):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.grid_map.header.stamp
        marker.ns = 'planner_markers'
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 2.5
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 5.0
        marker.color.a = 1.0  # Alpha channel for visibility
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        
        if marker_id == 0:
            self.pub_start.publish(marker)
        elif marker_id == 1:
            self.pub_goal.publish(marker)

    def convert_path_to_world_coordinates(self, path, map_length_x, y_offset, resolution):
        world_coordinates = []
        for cell in path:
            x_world = float(-cell[0]*resolution + map_length_x) # Convert from grid to world coordinates (Inverse of conversion to grid)
            y_world = float((-cell[1] + y_offset)*resolution) 
            world_coordinates.append((x_world, y_world))
        return world_coordinates

    def fomulate_path(self, path, time_stamp):
        path_msgs = Path()
        path_msgs.header.frame_id = 'base_link'
        path_msgs.header.stamp = time_stamp
        poses = []
        for location in path:
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.header.stamp = time_stamp

            # Location
            pose.pose.position.x = location[0]
            pose.pose.position.y = location[1]
            pose.pose.position.z = 1.0

            # Orientation
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            
            poses.append(pose)
        path_msgs.poses = poses
        return path_msgs

    # Function for handling the goal request
    async def execute_callback(self, goal_handle):
        self.get_logger().info('Received navigation goal request.')

        self.target_latitude = goal_handle.request.latitude
        self.target_longitude = goal_handle.request.longitude
        self.threshold = goal_handle.request.threshold

        self.get_logger().info(f'Navigating to GPS: {self.target_latitude}, {self.target_longitude} with threshold {self.threshold} meters.')

        # Navigation loop
        rate = self.create_rate(2) # 10Hz
        while rclpy.ok():
            # 1) Check if location, heading and costmap are available
            if self.current_latitude is None or self.current_longitude is None or self.current_yaw is None: # is None or self.grid_map
                self.get_logger().error(f'Current position is {self.current_latitude is None} and {self.current_longitude is None}, heading is {self.current_yaw is None} and gridmap is {self.grid_map is None}. Waiting for data...')
                continue
            rate.sleep()  # Sleep to prevent busy waiting

            # Calculate distance to goal
            distance, bearing = calculate_heading_and_distance(self.current_latitude, self.current_longitude,
                                                               self.target_latitude, self.target_longitude)
            self.get_logger().info(f'Current distance to goal: {distance} meters with bearing {bearing} radians.')
            
            # 2) Check if within threshold
            if distance <= self.threshold:
                self.get_logger().info('Reached the goal within threshold.')
                goal_handle.succeed()
                result = NavigateToGPS.Result()
                result.success = True
                return result

            # 3) Calculate angle of goal relative to current heading
            angle_to_goal = bearing - self.current_yaw  # Counter-clockwise positive (alignes with Z in ENU frame)
            goal_pos_x_raw, goal_pos_y_raw = calculate_position(angle_to_goal, distance)

            # 4) Derive cost_map information
            map_resolution = self.grid_map.info.resolution
            map_length_x = self.grid_map.info.length_x
            map_length_y = self.grid_map.info.length_y
            height_grid = grid_map_to_numpy(self.grid_map, 'elevation')  # Replace NaN values with 100 to indicate non-traversable areas
            cid_grid = grid_map_to_numpy(self.grid_map, 'class_id', fill_nan=True, nan_value=100)  # Replace NaN values with 100 to indicate non-traversable areas

            # 5) Check if goal is within bounds of costmap and correct if necessary
            min_x, max_x = 0, map_length_x
            min_y, max_y = -map_length_y/2, map_length_y/2
            goal_x, goal_y = 0, 0

            if not min_x < goal_pos_x_raw < max_x or not min_y < goal_pos_y_raw < max_y:
                if goal_pos_x_raw < 0 and goal_pos_y_raw < 0:
                    self.get_logger().info('Goal is behind the robot on the right side. Tight turn to the right must be performed')
                    goal_x, goal_y = 1.25, -1.25
                # Correct the position from outside to inside bounds
                elif goal_pos_x_raw < 0 and goal_pos_y_raw > 0:
                    self.get_logger().info('Goal is behind the robot on the left side. Tight turn to the left must be performed')
                    goal_x, goal_y = 1.25, 1.25
                else:
                    goal_x, goal_y = bring_goal_in_bounds(map_length_x, map_length_y, angle_to_goal, distance)
                    goal_x, goal_y = goal_x*0.75, goal_y*0.75  # Bring the goal a bit closer to the robot to prevent issues with path planning at the edges of the map
                    self.get_logger().info(f'Goal is rectified to x: {goal_x} and y: {goal_y}')
            else:
                goal_x, goal_y = goal_pos_x_raw, goal_pos_y_raw
                self.get_logger().info(f'Goal is in bounds at x: {goal_x} and y: {goal_y}')

            
            self.publish_marker(np.floor(goal_x/map_resolution), 
                                np.floor(goal_y/map_resolution), 
                                1, 
                                (1.0, 0.0, 0.0))  # Publish goal marker in red
            self.publish_marker(7.0, 0.0, 0, (0.0, 1.0, 0.0))  # Publish start marker in green

            # 6) Convert goal position to grid cell indices
            y_offset = int(map_length_y / (2 * map_resolution))  # Shift values to positive range
            x_b = np.floor((map_length_x - goal_x)/map_resolution).astype(np.int32)
            y_b = np.floor(-goal_y/map_resolution).astype(np.int32) + y_offset

            # 7) Run A* algorithm to find path
            # Costlist order: [background, smooth, rough, bumpy, forbidden, obstacle]
            # """
            path = calculate_path(start_cell = (int(map_length_x)-7, int(map_length_y/2)), 
                                goal_cell = (x_b, y_b), 
                                height_grid = height_grid,
                                cid_grid = cid_grid,
                                cost_list = [100, 10, 20, 30, 40, 75], 
                                resolution = map_resolution)
            
            if path is None:
                BLUE = '\033[94m'
                RESET = '\033[0m'
                self.get_logger().info(f'{BLUE}Failed to find a path to the goal.{RESET}')
                goal_handle.abort()
                result = NavigateToGPS.Result()
                result.success = False
                return result
            else:
                GREEN = '\033[92m'
                RESET = '\033[0m'
                self.get_logger().info(f'{GREEN}path found with {len(path)} waypoints. Publishing path and continuing navigation.{RESET}')
                # Create ROS2 Path message
                path_world_coordinates = self.convert_path_to_world_coordinates(path, map_length_x, y_offset, map_resolution)
                path_msg = self.fomulate_path(path_world_coordinates, self.grid_map.header.stamp)
                self.pub_path.publish(path_msg)
                # """

            feedback = NavigateToGPS.Feedback()
            feedback.distance_to_goal = distance
            goal_handle.publish_feedback(feedback)

def main(args=None):
    rclpy.init()
    planner_server_node = PlannerServer()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(planner_server_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        planner_server_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()