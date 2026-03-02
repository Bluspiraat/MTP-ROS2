import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from mtp_planner_interfaces.action import NavigateToGPS  # Import custom action

class PlannerClient(Node):
    def __init__(self, waypoint_list):
        super().__init__('planner_client_node')
        self._action_client = ActionClient(self, NavigateToGPS, 'navigate_gps')
        self.waypoint_list = waypoint_list
        self.failed_attemps = 0
        self.max_attempts = 100
        self.target_lat = None
        self.target_long = None
        self.threshold_distance = 5.0 # Meters

        self.get_logger().info(f'Planner Client Node with {len(waypoint_list)} waypoints has been started.')

        # Start the missions
        self.send_next_waypoint()

    def send_next_waypoint(self):
        # Check if there are more waypoints
        if not self.waypoint_list:
            self.get_logger().info('All waypoints have been sent.')
            return

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available for the past 5 seconds!')
            return
    
        new_wp = self.waypoint_list.pop(0)
        self.target_lat = new_wp[0]
        self.target_long = new_wp[1]
        print(f'New waypoint to send: {new_wp}')

        goal_msg = NavigateToGPS.Goal()
        goal_msg.latitude = new_wp[0]
        goal_msg.longitude = new_wp[1]
        goal_msg.threshold = self.threshold_distance  # meters

        self.get_logger().info(f'Sending waypoint: Lat {new_wp[0]}, Lon {new_wp[1]}')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    # Listen to the the response on the sent goal
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by the planner server.')
            return

        self.get_logger().info('Goal accepted, travelling to sent waypoint.')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    # Listen to the feedback of the planner.
    def feedback_callback(self, feedback_msg):
        dist = feedback_msg.feedback.distance_to_goal
        self.get_logger().info(f'Current distance to goal: {dist:.2f} meters', throttle_duration_sec=2.0)

    # Listen to the result of the planner.
    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if result.success:
            GREEN = '\033[92m'
            RESET = '\033[0m'
            self.get_logger().info(f'{GREEN}Successfully reached the waypoint!{RESET}')
        else:
            self.get_logger().error('Failed to reach the waypoint.')
            self.failed_attemps += 1
            if self.failed_attemps < self.max_attempts:
                self.get_logger().info('Retrying to send the same waypoint.')
                self.waypoint_list.insert(0, [self.target_lat, self.target_long])

        # Send the next waypoint
        self.send_next_waypoint()


def main(args=None):
    # Triangle on pleintje:
    coords_pleintje = [
        [52.246206, 6.846737],
        [52.246118, 6.846550],
        [52.246146, 6.846487],
        [52.246146, 6.846487],
        [52.246248, 6.846520],
        [52.246216, 6.846707]
    ]

    coords_grass = [
        [52.24607064801895, 6.847182684447783],
        [52.245998389392874, 6.847406648908608],
        [52.246034108160636, 6.847550147092346],
        [52.246085767733405, 6.84732341202871],
        [52.246110052940814, 6.84718713750989]
    ]

    house_loop = [
        [52.246257799373886, 6.84724632672126],
        [52.24645281414151, 6.847236938988458],
        [52.24651645035341, 6.846894286784699],
        [52.24643970490488, 6.846585800213233],
        [52.246243868966175, 6.846648161574737],
        [52.24617243174153, 6.846785624789298]
    ]

    rclpy.init(args=args)
    planner_client_node = PlannerClient(house_loop)
    rclpy.spin(planner_client_node)
    planner_client_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()