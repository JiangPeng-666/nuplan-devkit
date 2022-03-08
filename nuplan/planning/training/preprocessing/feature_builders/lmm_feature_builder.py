from __future__ import annotations

from typing import Dict, List, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint,StateSE2
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder, \
    AbstractModelFeature, FeatureBuilderMetaData
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.raster_utils import get_agents_raster, get_baseline_paths_raster, get_baseline_paths_raster_with_filter, \
    get_ego_raster, get_roadmap_raster, get_ego_with_past_raster


import cv2

from nuplan.planning.training.visualization.raster_visualization import Color

class LMMFeatureBuilder(AbstractFeatureBuilder):
    """
    LMM builder responsible for constructing model input features.
    """

    def __init__(
            self,
            map_features: Dict[str, int],
            target_width: int,
            target_height: int,
            target_pixel_size: float,
            ego_width: float,
            ego_front_length: float,
            ego_rear_length: float,
            ego_longitudinal_offset: float,
            baseline_path_thickness: int,
            trajectory_sampling: TrajectorySampling,
            use_trafficlight: bool = False,
            use_rgb: bool = True,
    ) -> None:
        """
        Initializes the class.

        :param map_features: name of map features to be drawn and their color for encoding.
        :param target_width: [pixels] target width of the raster
        :param target_height: [pixels] target height of the raster
        :param target_pixel_size: [m] target pixel size in meters
        :param ego_width: [m] width of the ego vehicle
        :param ego_front_length: [m] distance between the rear axle and the front bumper
        :param ego_rear_length: [m] distance between the rear axle and the rear bumper
        :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
                                        0.0 means place the ego at 1/2 from the bottom of the raster image.
                                        0.25 means place the ego at 1/4 from the bottom of the raster image.
        :param baseline_path_thickness: [pixels] the thickness of baseline paths in the baseline_paths_raster.
        """
        self.map_features = map_features
        self.target_width = target_width
        self.target_height = target_height
        self.target_pixel_size = target_pixel_size

        self.ego_longitudinal_offset = ego_longitudinal_offset
        self.baseline_path_thickness = baseline_path_thickness
        self.raster_shape = (self.target_width, self.target_height)

        x_size = self.target_width * self.target_pixel_size / 2.0
        y_size = self.target_height * self.target_pixel_size / 2.0
        x_offset = 2.0 * self.ego_longitudinal_offset * x_size
        self.x_range = (-x_size + x_offset, x_size + x_offset)
        self.y_range = (-y_size, y_size)

        self.ego_width_pixels = int(ego_width / self.target_pixel_size)
        self.ego_front_length_pixels = int(ego_front_length / self.target_pixel_size)
        self.ego_rear_length_pixels = int(ego_rear_length / self.target_pixel_size)

        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon

        self.use_trafficlight = use_trafficlight
        self.use_rgb = use_rgb

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """ Inherited, see superclass. """
        return "raster"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """ Inherited, see superclass. """
        return Raster  # type: ignore

    def get_features_from_scenario(self, scenario: AbstractScenario) -> Raster:
        """ Inherited, see superclass. """
        map_api = scenario.map_api
        # Retrieve present/past ego states and agent boxes
        anchor_ego_state = scenario.initial_ego_state

        past_ego_states = scenario.get_ego_past_trajectory(iteration=0,
                                                           num_samples=self.num_past_poses,
                                                           time_horizon=self.past_time_horizon)
        sampled_past_ego_states = past_ego_states + [anchor_ego_state]
        time_stamps = scenario.get_past_timestamps(iteration=0,
                                                   num_samples=self.num_past_poses,
                                                   time_horizon=self.past_time_horizon) + [scenario.start_time]
        # Retrieve present/future agent boxes
        present_agent_boxes = scenario.initial_detections.boxes
        past_agent_boxes = [detection.boxes
                            for detection in scenario.get_past_detections(iteration=0,
                                                                          num_samples=self.num_past_poses,
                                                                          time_horizon=self.past_time_horizon)]

        # Extract and pad features
        sampled_past_observations = past_agent_boxes + [present_agent_boxes]

        # Get trafficlight info
        trafficlight: List[TrafficLightStatusData] = scenario.get_traffic_light_status_at_iteration(iteration=0)
        trafficlight_dict = {tl_statustype:[] for tl_statustype in TrafficLightStatusType}
        for tl in trafficlight:
            trafficlight_dict[tl.status].append(tl.lane_connector_id)
        
        if len(trafficlight_dict[TrafficLightStatusType['RED']])>0:
            print(2)

        assert len(sampled_past_ego_states) == len(sampled_past_observations), \
            "Expected the trajectory length of ego and agent to be equal. " \
            f"Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}"

        assert len(sampled_past_observations) > 2, "Trajectory of length of " \
                                                   f"{len(sampled_past_observations)} needs to be at least 3"

        return self._compute_feature(sampled_past_ego_states, sampled_past_observations, time_stamps, map_api, trafficlight_dict)

    def get_features_from_simulation(self, ego_states: List[EgoState], observations: List[Observation],
                                     meta_data: FeatureBuilderMetaData) -> Raster:
        """ Inherited, see superclass. """
        ego_state = ego_states[-1]
        observation = observations[-1]

        if isinstance(observation, Detections):
            return self._compute_feature(ego_state, observation, meta_data.map_api)
        else:
            raise TypeError(f"Observation was type {observation.detection_type()}. Expected Detections")

    def _compute_feature(self, sampled_past_ego_states: List[EgoState],
                         sampled_past_observations: List[List[Box3D]],
                         time_stamps: List[TimePoint],
                         map_api: AbstractMap,
                         trafficlight_dict: Dict[TrafficLightStatusType, List] = None,) -> Raster:

        # The last frame is the current frame, set as archor.
        anchor_ego_state = sampled_past_ego_states[-1]

        # Check if sampled_past_observations and sampled_past_ego_states has the same length
        assert len(sampled_past_observations) == len(sampled_past_ego_states), \
            "Expected sampled_past_observations and sampled_past_ego_states to have the same length, \
            But got {} and {}".format(len(sampled_past_observations), len(sampled_past_ego_states))
        
        # Raster Agents and Ego
        sampled_past_num = len(sampled_past_observations)
        agents_raster_list = np.zeros((sampled_past_num, self.raster_shape[0],self.raster_shape[1]), dtype=np.float32)
        ego_raster_list = np.zeros((sampled_past_num, self.raster_shape[0],self.raster_shape[1]), dtype=np.float32)

        # For each frame, start from the earliest to current
        for i in range(sampled_past_num):

            agents_raster_list[i,:,:] = get_agents_raster(
                anchor_ego_state,
                Detections(boxes = sampled_past_observations[i]),
                self.x_range,
                self.y_range,
                self.raster_shape,
            )

            ego_raster_list[i,:,:] = get_ego_with_past_raster(
                anchor_ego_state,
                sampled_past_ego_states[i],
                self.x_range,
                self.y_range,
                self.raster_shape,
            )
        ego_raster = self.merge_with_fade(ego_raster_list)
        agents_raster = self.merge_with_fade(agents_raster_list)

        # Raster Roadmap
        roadmap_raster = get_roadmap_raster(
            anchor_ego_state,
            map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
        )

        # Raster Baseline Path
        if self.use_trafficlight == True:
            baseline_paths_rasters = [np.zeros((self.raster_shape[0],self.raster_shape[1]), dtype=np.float32) for i in range(len(TrafficLightStatusType))]
            for i, trafficlight_status in enumerate(TrafficLightStatusType):
                filter = trafficlight_dict[trafficlight_status]
                if len(filter)>0 and trafficlight_status == TrafficLightStatusType['RED']:
                    print(1)
                else:
                    continue
                if len(filter) == 0:
                    continue
                baseline_paths_rasters[i] = get_baseline_paths_raster_with_filter(
                    anchor_ego_state,
                    map_api,
                    self.x_range,
                    self.y_range,
                    self.raster_shape,
                    self.target_pixel_size,
                    filter,
                    self.baseline_path_thickness,
                )
            baseline_paths_raster = get_baseline_paths_raster(
                anchor_ego_state,
                map_api,
                self.x_range,
                self.y_range,
                self.raster_shape,
                self.target_pixel_size,
                self.baseline_path_thickness,
            )
            collated_layers = np.dstack([ego_raster, agents_raster, roadmap_raster, baseline_paths_raster
                            ]+baseline_paths_rasters).astype(np.float32)
        else:
            baseline_paths_raster = get_baseline_paths_raster(
                anchor_ego_state,
                map_api,
                self.x_range,
                self.y_range,
                self.raster_shape,
                self.target_pixel_size,
                self.baseline_path_thickness,
            )
            collated_layers = np.dstack([ego_raster, agents_raster, roadmap_raster,
                            baseline_paths_raster]).astype(np.float32)
        
        if self.use_rgb:
            res_image = self.to_rgb(
                roadmap_raster=roadmap_raster,
                agents_raster = agents_raster,
                ego_raster = ego_raster,
                baseline_paths_raster = baseline_paths_raster
                )
        else:
            res_image = collated_layers
        
        #cv2.imwrite("/home/fla/nuplan-devkit/sample/test00.png", res_image)
        # cv2.imwrite("/home/fla/nuplan-devkit/sample/test11.png", ego_raster)
        # cv2.imwrite("/home/fla/nuplan-devkit/sample/test22.png", agents_raster)
        # cv2.imwrite("/home/fla/nuplan-devkit/sample/test33.png", roadmap_raster)
        # cv2.imwrite("/home/fla/nuplan-devkit/sample/test44.png", baseline_paths_raster)
        # cv2.imwrite("/home/fla/nuplan-devkit/sample/test55.png", baseline_paths_rasters[0])
        # cv2.imwrite("/home/fla/nuplan-devkit/sample/test66.png", baseline_paths_rasters[1])
        # cv2.imwrite("/home/fla/nuplan-devkit/sample/test77.png", baseline_paths_rasters[2])

        return Raster(data=res_image)

    def merge_with_fade(self, images: List[npt.NDArray[np.float32]]):
        fade_rate = 10
        images_num = len(images)
        assert images_num>0, "Expected input images to be more than 0."
        
        if images_num ==1:
            return images[0]
        else:
            res_image = np.zeros(images[0].shape)
        pixel_value = 256-fade_rate*images_num # From 255-(images_num-1) to 255-0
        for image in images:
            res_image=np.where(image!=0, image*pixel_value, res_image)
            pixel_value+=fade_rate
        return res_image

    def to_rgb(self,
        roadmap_raster: npt.NDArray[np.float32],
        agents_raster: npt.NDArray[np.float32],
        ego_raster: npt.NDArray[np.float32],
        baseline_paths_raster: npt.NDArray[np.float32],
        baseline_paths_rasters: npt.NDArray[np.float32] = None,
    ):

        if ego_raster.max()<1:
            ego_raster*=255
        if roadmap_raster.max()<1:
            roadmap_raster*=255
        if agents_raster.max()<1:
            agents_raster*=255
        if baseline_paths_raster.max()<1:
            baseline_paths_raster*=255
        
        res_image = np.ones((self.raster_shape[0], self.raster_shape[1], 3), dtype=np.float32)
        res_image[:,:] = np.array(Color.BACKGROUND.value)

        for i in range(self.raster_shape[0]):
            for j in range(self.raster_shape[1]):

                roadmap_pixel = roadmap_raster[i,j]
                agents_pixel = agents_raster[i,j]
                ego_pixel = ego_raster[i,j]
                baseline_paths_pixel = baseline_paths_raster[i,j]
                
                if ego_pixel>0:
                    res_image[i,j,:] = np.array(Color.EGO.value)*ego_pixel
                elif agents_pixel>0:
                    res_image[i,j,:] = np.array(Color.AGENTS.value)*agents_pixel
                elif baseline_paths_pixel>0:
                    res_image[i,j,:] = np.array(Color.BASELINE_PATHS.value)*baseline_paths_pixel
                elif roadmap_pixel>0:
                    res_image[i,j,:] = np.array(Color.ROADMAP.value)*roadmap_pixel

        return res_image