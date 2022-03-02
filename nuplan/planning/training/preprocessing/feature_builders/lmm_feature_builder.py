from __future__ import annotations

from typing import Dict, List, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint,StateSE2
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder, \
    AbstractModelFeature, FeatureBuilderMetaData
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.raster_utils import get_agents_raster, get_baseline_paths_raster, \
    get_ego_raster, get_roadmap_raster
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import build_ego_features, \
    compute_yaw_rate_from_states, extract_and_pad_agent_poses, extract_and_pad_agent_sizes, \
    extract_and_pad_agent_velocities, filter_agents
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses, \
    convert_absolute_to_relative_velocities
from nuplan.planning.training.preprocessing.features.agents import AgentsFeature

class LMMFeatureBuilder(AbstractFeatureBuilder):
    """
    LMM builder responsible for constructing model input features.
    """

    def __init__(
            self,
            map_features: Dict[str, int],
            num_input_channels: int,
            target_width: int,
            target_height: int,
            target_pixel_size: float,
            ego_width: float,
            ego_front_length: float,
            ego_rear_length: float,
            ego_longitudinal_offset: float,
            baseline_path_thickness: int,
            trajectory_sampling: TrajectorySampling,
    ) -> None:
        """
        Initializes the class.

        :param map_features: name of map features to be drawn and their color for encoding.
        :param num_input_channels: number of input channel of the raster model.
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
        self.num_input_channels = num_input_channels
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

        assert len(sampled_past_ego_states) == len(sampled_past_observations), \
            "Expected the trajectory length of ego and agent to be equal. " \
            f"Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}"

        assert len(sampled_past_observations) > 2, "Trajectory of length of " \
                                                   f"{len(sampled_past_observations)} needs to be at least 3"

        return self._compute_feature(sampled_past_ego_states, sampled_past_observations, time_stamps, map_api)

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
                         map_api: AbstractMap) -> Raster:

        # The last frame is the current frame, set as archor.
        anchor_ego_state = sampled_past_ego_states[-1]

        # Check if sampled_past_observations and sampled_past_ego_states has the same length
        assert len(sampled_past_observations) == len(sampled_past_ego_states), \
            "Expected sampled_past_observations and sampled_past_ego_states to have the same length, \
            But got {} and {}".format(len(sampled_past_observations), len(sampled_past_ego_states))

        sampled_past_num = len(sampled_past_observations)

        agents_raster_list = np.zeros((sampled_past_num, self.raster_shape[0],self.raster_shape[1]), dtype=np.float32)
        ego_raster_list = np.zeros((sampled_past_num, self.raster_shape[0],self.raster_shape[1]), dtype=np.float32)

        # For each frame, start from the earliest to current
        for i in range(sampled_past_num):

            agents_raster_list[i,:,:] = get_agents_raster(
                sampled_past_ego_states[i],
                sampled_past_observations[i],
                self.x_range,
                self.y_range,
                self.raster_shape,
            )

            ego_raster_list[i,:,:] = get_ego_raster(
                self.raster_shape,
                self.ego_longitudinal_offset,
                self.ego_width_pixels,
                self.ego_front_length_pixels,
                self.ego_rear_length_pixels,
            )

        roadmap_raster = get_roadmap_raster(
            anchor_ego_state,
            map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
        )
        baseline_paths_raster = get_baseline_paths_raster(
            anchor_ego_state,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            self.baseline_path_thickness
        )

        collated_layers = np.dstack([ego_raster, agents_raster, roadmap_raster,  # type: ignore
                                     baseline_paths_raster]).astype(np.float32)

        # Ensures the last channel is the number of channel.
        if collated_layers.shape[-1] != self.num_input_channels:
            raise RuntimeError(
                f'Invalid raster numpy array. '
                f'Expected {self.num_input_channels} channels, got {collated_layers.shape[-1]} '
                f'Shape is {collated_layers.shape}')

        return Raster(data=collated_layers)

    def merge_with_fade(self, images: List[npt.NDArray[np.float32]]):
        images_num = len(images)
        assert images_num>0, "Expected input images to be more than 0."
        
        if images_num ==1:
            return images[0]
        else:
            res_image = np.zeros(images[0].shape)
        pixel_value = 256-images_num # From 255-(images_num-1) to 255-0
        for image in images:
            res_image=np.where(image!=0, image*pixel_value, res_image)
            pixel_value+=1
        return res_image

