from __future__ import annotations

from typing import Dict, List, Optional, Set, Type, cast

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import LaneSegmentConnections, LaneSegmentCoords
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder, \
    FeatureBuilderMetaData
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.common.maps.abstract_map import SemanticMapLayer
import math
import numpy as np
from matplotlib import pyplot as plt
import datetime


def _transform_to_relative_frame(coords: npt.NDArray[np.float32],
                                 anchor_state: StateSE2) -> npt.NDArray[np.float32]:
    """
    Transform a set of coordinates to the the given frame.
    :param coords: <np.ndarray: num_coords, 2> Coordinates to be transformed.
    :param anchor_state: The coordinate frame to transform to.
    :return: <np.ndarray: num_coords, 2> Transformed coordinates.
    """
    # Extract transform
    transform = np.linalg.inv(anchor_state.as_matrix())  # type: ignore
    # Homogenous Coordinates
    coords = np.pad(coords, ((0, 0), (0, 1)), 'constant', constant_values=1.0)  # type: ignore
    coords = transform @ coords.transpose()
    return cast(npt.NDArray[np.float32], coords.transpose()[:, :2])


def _accumulate_connections(node_idx_to_neighbor_dict: Dict[int, Dict[str, Set[int]]], scales: List[int]) \
        -> Dict[int, npt.NDArray[np.float32]]:
    """
    Accumulate the connections over multiple scales
    :param node_idx_to_neighbor_dict: {node_idx: neighbor_dict} where each neighbor_dict
                                      will have format {'i_hop_neighbors': set_of_i_hop_neighbors}
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
    """
    # Get the connections of each scale.
    multi_scale_connections = {}
    for scale in scales:
        scale_connections = []
        for node_idx, neighbor_dict in node_idx_to_neighbor_dict.items():
            for n_hop_neighbor in neighbor_dict[f"{scale}_hop_neighbors"]:
                scale_connections.append([node_idx, n_hop_neighbor])

        # if cannot find n-hop neighbors, return empty connection with size [0,2]
        if len(scale_connections) == 0:
            scale_connections = np.empty([0, 2], dtype=np.int64)  # type: ignore

        multi_scale_connections[scale] = np.array(scale_connections)

    return multi_scale_connections


def _generate_multi_scale_connections(connections: npt.NDArray[np.int32],
                                      scales: List[int]) -> Dict[int, npt.NDArray[np.float32]]:
    """
    Generate multi-scale connections by finding the neighbors up to max(scales) hops away for each node.
    :param connections: <np.ndarray: num_connections, 2>. A 1-hop connection is represented by [start_idx, end_idx]
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
             Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
    """
    # This dict will have format {node_idx: neighbor_dict},
    # where each neighbor_dict will have format {'i_hop_neighbors': set_of_i_hop_neighbors}.
    node_idx_to_neighbor_dict: Dict[int, Dict[str, Set[int]]] = {}

    # Initialize the data structure for each node with its 1-hop neighbors.
    for connection in connections:
        start_idx, end_idx = list(connection)
        if start_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[start_idx] = {"1_hop_neighbors": set()}
        if end_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[end_idx] = {"1_hop_neighbors": set()}
        node_idx_to_neighbor_dict[start_idx]["1_hop_neighbors"].add(end_idx)

    # Find the neighbors up to max(scales) hops away for each node.
    for scale in range(2, max(scales) + 1):
        for neighbor_dict in node_idx_to_neighbor_dict.values():
            neighbor_dict[f"{scale}_hop_neighbors"] = set()
            for n_hop_neighbor in neighbor_dict[f"{scale - 1}_hop_neighbors"]:
                for n_plus_1_hop_neighbor in node_idx_to_neighbor_dict[n_hop_neighbor]["1_hop_neighbors"]:
                    neighbor_dict[f"{scale}_hop_neighbors"].add(n_plus_1_hop_neighbor)

    return _accumulate_connections(node_idx_to_neighbor_dict, scales)


class VectorMapFeatureBuilder(AbstractFeatureBuilder):
    """
    Vector map feature builder. Note: this builder uses the feature builder that is used for LSN
    """

    def __init__(self, radius: float, connection_scales: Optional[List[int]] = None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        self._radius = radius
        self._connection_scales = connection_scales

    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """ Inherited, see superclass. """
        return VectorMap  # type: ignore

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """ Inherited, see superclass. """
        return "vector_map"

    @classmethod
    def get_features_from_scenario_for_epsilon(self, scenario: AbstractScenario) -> List:
        """ get features for epsilon, nums *  7, id, length, points, father_id, child_id, left_id, right_id """

        def judge_parallel(line1, line2):
            """to judge whether the two lines are parallel"""
            if abs(line1[0] * line2[1] - line1[1] * line2[0]) < 0.5:
                return True
            else:
                return False

        def judge_line_or_circle(points) -> bool:
            """to judge the points input is on a line or a circle"""
            pt_num = len(points)
            i, j = 0, pt_num - 1
            pt_1, pt_2 = np.array(points[0]), np.array(points[pt_num - 1])
            cur = np.divide(pt_2 - pt_1, np.linalg.norm(pt_2 - pt_1))

            while i < j:
                pt_1, pt_2 = np.array(points[i]), np.array(points[j])
                tmp = np.divide(pt_2 - pt_1, np.linalg.norm(pt_2 - pt_1))
                if not judge_parallel(cur, tmp):
                    return False

                i += 1
                j -= 1
            return True
        
        def __line_magnitude(x1, y1, x2, y2):
            """get the distance between two points on a line segment"""
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        def get_projected_position(point, line) -> bool:
            """to determine whether the projected point of the point in the direction of 
            the line segment is on the line segment"""
            px, py = point
            x1, y1 = line[0]
            x2, y2 = line[1]
            line_magnitude = __line_magnitude(x1, y1, x2, y2)
            u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
            u = u1 / (line_magnitude * line_magnitude)
            if u < 1e-5 or u > 1:
                return False
            else: 
                return True

        def get_dist(point, line):
            """get the distance of the point to the line segment"""
            px, py = point
            x1, y1 = line[0]
            x2, y2 = line[1]
            line_magnitude = __line_magnitude(x1, y1, x2, y2)
            u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
            u = u1 / (line_magnitude * line_magnitude)
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)

            return distance

        def get_relative_position(point, line) ->bool:
            """to determine whether the point is on the left or right of the line segment(a->b)"""
            a, b = line[0], line[1]

            tmp = np.cross([a[0] - point[0], a[1] - point[1]], [b[0] - point[0], b[1] - point[1]])
            if (tmp > 0):
                return True # left
            else: 
                return False # right

        # for debugging, here we take only the objects around ego vehicle
        lane_coords_epsi = []
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)

        # for debug
        lanes_connector_debug_list, lanes_debug_list = [], []
        lanes_connector_debug = scenario.map_api.get_proximal_map_objects(ego_coords, 50, [SemanticMapLayer.LANE_CONNECTOR])
        tmp_1 = []
        for _, tmp_2 in lanes_connector_debug.items():
            tmp_1.extend(tmp_2)
        for i in range(len(tmp_1)):
            lanes_connector_debug_list.append(int(tmp_1[i].id))

        lanes_debug = scenario.map_api.get_proximal_map_objects(ego_coords, 50, [SemanticMapLayer.LANE])
        tmp_1 = []
        for _, tmp_2 in lanes_debug.items():
            tmp_1.extend(tmp_2)
        for i in range(len(tmp_1)):
            lanes_debug_list.append(int(tmp_1[i].id))

        map_objects_integrated = scenario.map_api.get_proximal_map_objects(ego_coords, 50, [SemanticMapLayer.LANE,
                                                                                      SemanticMapLayer.LANE_CONNECTOR])
                                                                                    
        for _, map_objects in map_objects_integrated.items():
            lane_coords_epsi.extend(map_objects)

        # the coords for the lanes to be used in EPSILON
        # 1.067584 is the unique identifier for epsilon to get all the objects
        # lane_coords_epsi = scenario.map_api.get_all_map_objects(Point2D(-1.067584, -1.067584), SemanticMapLayer.LANE)
        # lane_coords_epsi.extend(scenario.map_api.get_all_map_objects(Point2D(-1.067584, -1.067584), SemanticMapLayer.LANE_CONNECTOR))

        lanes_epsi_list = [] # nums *  7, id, length, points, father_id, child_id, left_id, right_id
        for i in range(len(lane_coords_epsi)):
            lane_tmp = []

            lane_epsi = lane_coords_epsi[i]
            lane_tmp.append(int(lane_epsi.id))

            lane_coords_epsi_blp = lane_epsi.baseline_path()
            lane_tmp.append(lane_coords_epsi_blp.linestring.length)

            lane_tmp_points = [[x, y] for x, y in zip(*lane_coords_epsi_blp.linestring.xy)]
            lane_tmp.append(lane_tmp_points)

            # lane -> lane_connector, lane_connector -> lane
            incoming_edges = lane_epsi.incoming_edges() # list 
            father_id = []
            for j in range(len(incoming_edges)):
                father_id.append(int(incoming_edges[j].id))
            lane_tmp.append(father_id)

            outgoing_edges = lane_epsi.outgoing_edges() # list
            child_id = []
            for j in range(len(outgoing_edges)):
                child_id.append(int(outgoing_edges[j].id))
            lane_tmp.append(child_id)

            lane_tmp.append(lane_epsi.polygon.area / lane_epsi.polygon.length) # width(tmp)
            lanes_epsi_list.append(lane_tmp)

        # sort the lanes_epsi_list based on the start x
        lanes_epsi_list.sort(key=lambda lane : lane[2][0][0])

        # to get the left and right lanes of each lane
        # 1. the directions of each lane and its neighbors should be the same
        # 2. the dist between the points of neighbor lanes should be close to the width
        #    of the polygon, here we take the multiplying of 3 as the threshold
        for i in range(len(lanes_epsi_list)):
            lane_neighbor_left, lane_neighbor_right  = [], []
            cur_dist = 1e5
            cur_lane_pts = lanes_epsi_list[i][2]
            for j in range(len(lanes_epsi_list)):
                if i == j: continue
                candidate_lane_num = len(lanes_epsi_list[j][2])
                candidate_lane_pts = lanes_epsi_list[j][2]
                candidate_line_flag = judge_line_or_circle(candidate_lane_pts)

                if candidate_line_flag: neighbor_threshold = 0.4
                else: neighbor_threshold = 0.2

                in_cnts = 0 # the num of points projected in the line segment
                left_cnts, right_cnts = 0, 0
                in_pts_dist = 0
                for k in range(len(candidate_lane_pts)):
                    project_flag = False
                    for pt_num in range(len(cur_lane_pts )- 1):
                        if project_flag: break
                        # if the projected point is in the lane segment, we sum the distance 
                        if get_projected_position(candidate_lane_pts[k], [cur_lane_pts[pt_num], cur_lane_pts[pt_num + 1]]):
                            in_cnts += 1 
                            in_pts_dist += get_dist(candidate_lane_pts[k], [cur_lane_pts[pt_num], cur_lane_pts[pt_num + 1]])
                            if get_relative_position(candidate_lane_pts[k], [cur_lane_pts[pt_num], cur_lane_pts[pt_num + 1]]):
                                left_cnts += 1
                            else: 
                                right_cnts += 1
                            project_flag = True
                                            
                if in_cnts > candidate_lane_num * neighbor_threshold:
                    if left_cnts * right_cnts != 0: continue # the two lanes are crossing
                    elif left_cnts: 
                        cur_dist_tmp = in_pts_dist / left_cnts
                        if cur_dist_tmp < lanes_epsi_list[i][-1] * 3:
                            if len(lane_neighbor_left) != 0:
                                if cur_dist_tmp > cur_dist:
                                    continue
                            if (lanes_epsi_list[j][0] < 0 or lanes_epsi_list[j][0] > 80000): continue
                            lane_neighbor_left = []
                            lane_neighbor_left.append(lanes_epsi_list[j][0])
                            cur_dist = cur_dist_tmp
                    else:
                        cur_dist_tmp = in_pts_dist / right_cnts
                        if cur_dist_tmp < lanes_epsi_list[i][-1] * 3:
                            if len(lane_neighbor_right) != 0:
                                if cur_dist_tmp > cur_dist:
                                    continue
                            if (lanes_epsi_list[j][0] < 0 or lanes_epsi_list[j][0] > 80000): continue
                            lane_neighbor_right = []
                            lane_neighbor_right.append(lanes_epsi_list[j][0])
                            cur_dist = cur_dist_tmp

            if len(lane_neighbor_left) == 0: lane_neighbor_left.append(-1)
            if len(lane_neighbor_right) == 0: lane_neighbor_right.append(-1)
            lanes_epsi_list[i] = lanes_epsi_list[i][:-1]
            lanes_epsi_list[i].append(lane_neighbor_left)
            lanes_epsi_list[i].append(lane_neighbor_right)
            print("Finish getting {}-th lanes".format(i))

        print("Finish getting lanes")

        # for debugging, visulize current lane here
        plt.figure()
        for i in range(len(lanes_epsi_list)):
            cur_lane_pts = lanes_epsi_list[i][2]
            dx, dy = [], []
            for j in range(len(cur_lane_pts)):
                dx.append(cur_lane_pts[j][0])
                dy.append(cur_lane_pts[j][1])

            plt.plot(dx, dy)
       
        lanes_id_list, lanes_length_list, points_list = [], [], []
        pre_connections_list, nxt_connections_list, left_connection_list, right_connection_list = [], [], [], []
        max_point_num, max_pre_num, max_nxt_num = 0, 0, 0
        for i in range(len(lanes_epsi_list)):
            lanes_id_list.append(lanes_epsi_list[i][:1][0]) # id
            lanes_length_list.append(lanes_epsi_list[i][1:2][0]) # length
            points_list.append(lanes_epsi_list[i][2:3][0]) # points
            max_point_num = max(max_point_num, len(points_list[-1]))
            
            pre_connections_list.append(lanes_epsi_list[i][3:4][0]) # father_id
            max_pre_num = max(max_pre_num, len(pre_connections_list[-1]))
            nxt_connections_list.append(lanes_epsi_list[i][4:5][0]) # child_id
            max_nxt_num = max(max_nxt_num, len(lanes_epsi_list[i][4:5][-1]))

            left_connection_list.append(lanes_epsi_list[i][5:6][0])  # left_id
            right_connection_list.append(lanes_epsi_list[i][6:][0])  # right_id

        for i in range(len(points_list)):
            if len(points_list[i]) < max_point_num:
                for _ in range(max_point_num - len(points_list[i])):
                    points_list[i].append([-1.0, -1.0])
            if len(pre_connections_list[i]) < max_pre_num:
                for _ in range(max_pre_num - len(pre_connections_list[i])):
                    pre_connections_list[i].append(-1)
            if len(nxt_connections_list[i]) < max_nxt_num:
                for _ in range(max_nxt_num - len(nxt_connections_list[i])):
                    nxt_connections_list[i].append(-1)
            
                    
        lanes_id, lanes_length, points = np.array(lanes_id_list), np.array(lanes_length_list), np.array(points_list)
        
        # taking only part of the lanes into consideration, here filter out those lanes that 
        # are not in chosen lanes
        lanes_to_filter = [pre_connections_list, nxt_connections_list, left_connection_list, right_connection_list]
        for i in range(len(lanes_to_filter)):
            for j in range(len(lanes_to_filter[i])):
                for k in range(len(lanes_to_filter[i][j])):
                    if lanes_to_filter[i][j][k] not in lanes_id:
                        lanes_to_filter[i][j][k] = -1

        pre_connections, nxt_connections = np.array(pre_connections_list), np.array(nxt_connections_list)
        left_connection, right_connection = np.array(left_connection_list), np.array(right_connection_list)

        res = list((lanes_id, lanes_length, points, pre_connections, nxt_connections, left_connection, right_connection,
        lanes_debug_list, lanes_connector_debug_list))

        return res

    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorMap:
        """ Inherited, see superclass. """

        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        ls_coords, ls_conns, ls_meta = scenario.map_api.get_neighbor_vector_map(ego_coords, self._radius)

        return self._compute_feature(ls_coords, ls_conns, ego_state.rear_axle)

    def get_features_from_simulation(self, ego_states: List[EgoState], observations: List[Observation],
                                     meta_data: FeatureBuilderMetaData) -> VectorMap:
        """ Inherited, see superclass. """

        ego_state = ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        ls_coords, ls_conns, ls_meta = meta_data.map_api.get_neighbor_vector_map(ego_coords, self._radius)

        return self._compute_feature(ls_coords, ls_conns, ego_state.rear_axle)

    def _compute_feature(self,
                         lane_coords: LaneSegmentCoords,
                         lane_conns: LaneSegmentConnections,
                         anchor_state: StateSE2) -> VectorMap:
        """
        :param lane_coords: A list of lane_segment coords in shape of [num_lane_segment, 2, 2]
        :param lane_conns: a List of lane_segment connections [start_idx, end_idx] in shape of [num_connection, 2]
        :return: VectorMap
        """

        lane_segment_coords = np.asarray(lane_coords.to_vector(), np.float32)
        lane_segment_conns = np.asarray(lane_conns.to_vector(), np.int64)

        # Transform the lane coordinates from global frame to ego vehicle frame.
        # Flatten lane_segment_coords from (num_lane_segment, 2, 2) to (num_lane_segment * 2, 2) for easier processing.
        lane_segment_coords = lane_segment_coords.reshape(-1, 2)
        lane_segment_coords = _transform_to_relative_frame(lane_segment_coords, anchor_state)
        lane_segment_coords = lane_segment_coords.reshape(-1, 2, 2).astype(np.float32)

        if self._connection_scales:
            # Generate multi-scale connections.
            multi_scale_connections = _generate_multi_scale_connections(
                lane_segment_conns, self._connection_scales
            )
        else:
            # Use the 1-hop connections if connection_scales is not specified.
            multi_scale_connections = {1: lane_segment_conns}

        return VectorMap(
            coords=[lane_segment_coords],
            multi_scale_connections=[multi_scale_connections],
        )
