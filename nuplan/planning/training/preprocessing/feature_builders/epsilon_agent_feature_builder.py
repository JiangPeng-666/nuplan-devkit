from typing import List, Type, cast

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder, \
    AbstractModelFeature, FeatureBuilderMetaData
from nuplan.planning.training.preprocessing.features.agents import AgentsFeature
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses, \
    convert_absolute_to_relative_velocities
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import build_ego_features, build_ego_full_features, \
    compute_yaw_rate_from_states, extract_and_pad_agent_poses, extract_and_pad_agent_sizes, \
    extract_and_pad_agent_velocities, filter_agents 


def _compute_feature(ego_history: List[EgoState],
                     agent_history: List[List[Box3D]],
                     time_stamps: List[TimePoint]) -> AgentsFeature:
    """
    This function is used to construct the feature during simulation
    :param ego_history: ego past trajectory comprising of EgoState
    :param agent_history: agent past trajectories [num_frames, num_agents]
    :param time_stamps: the time stamps of each frame
    :return: constructed features
    """

    anchor_ego_state = ego_history[0]
    delta_t = time_stamps[-1].time_s - time_stamps[-2].time_s 
    
    # agent_track_tokens = [box.track_token for box in agent_history[-1]]
    for box in agent_history[-1]:
        # for velocity
        last_box = [tmp_box for tmp_box in agent_history[-2] if tmp_box.track_token == box.track_token]
        if (len(last_box) == 0): continue
        # if box.length < 4 or box.width < 1: continue
        box.velocity[0] = (box.center[0] - last_box[0].center[0]) / delta_t
        box.velocity[1] = (box.center[1] - last_box[0].center[1]) / delta_t
        box.velocity[2] = 13.33 # in case of being padded


    agent_history = filter_agents(agent_history, reverse=True)
    if len(agent_history[-1]) == 0:
        # Return empty array when there are no agents in the scene
        agent_features = np.empty(
            shape=(len(agent_history), 0, AgentsFeature.agents_states_dim()), dtype=np.float32)
    else:
        # 5 * agents_nums, poses denoted by StateSE2(x, y, heading)
        agent_states_horizon = extract_and_pad_agent_poses(agent_history, reverse=True)
        # 5 * agents_nums, (width, length)
        agent_sizes_horizon = extract_and_pad_agent_sizes(agent_history, reverse=True)
        # 5 * agents_nums, velocities denoted by StateSE2(x, y, yaw)
        agent_velocities_horizon = extract_and_pad_agent_velocities(agent_history, reverse=True)

        # Get all poses relative to the ego coordinate system
        agent_relative_poses = [convert_absolute_to_relative_poses(anchor_ego_state.rear_axle, states)
                                for states in agent_states_horizon]

        agent_relative_velocities = [
            convert_absolute_to_relative_velocities(StateSE2(anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                                                             anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
                                                             anchor_ego_state.rear_axle.heading), states)
            for states in agent_velocities_horizon]

        absolute_velocities = (np.asarray([state.serialize() for state in agent_velocities_horizon[-1]])).astype(np.float32)
        absolute_states = (np.asarray([state.serialize() for state in agent_states_horizon[-1]])).astype(np.float32)
        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_states(agent_states_horizon, time_stamps)

        # Append the agent box pose, velocities, sizes  together for epsilon
        agent_features_list = [np.hstack([poses, velocities, sizes])  # type:ignore
                               for poses, velocities, sizes in
                               zip(absolute_states, absolute_velocities, agent_sizes_horizon[-1])]


        # # Append the agent box pose, velocities  together
        # agent_features_list = [np.hstack([poses, velocities, np.expand_dims(yaw_rate, axis=1), sizes])  # type:ignore
        #                        for poses, velocities, yaw_rate, sizes in
        #                        zip(agent_relative_poses, agent_relative_velocities, yaw_rate_horizon.transpose(),
        #                            agent_sizes_horizon)]
        # agent_features = np.stack(agent_features_list)
        
        # to meet the format requirements， the latest agents_feature_list(nums * 8) is stacked twice
        # the agent_feature below is (2 * nums * 8)
        agent_features = np.stack([agent_features_list, agent_features_list], axis = 0)

    # ego_features = build_ego_features(ego_history, reverse=True)

    # for epsilon, ego absolute poses, 1*3, x,y,heading
    ego_features = build_ego_full_features(ego_history, reverse=True) 

    # agents, nums * 8
    return AgentsFeature(ego=[ego_features], agents=[agent_features])


class AgentsFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes AgentsFeatureBuilder.

        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """ Inherited, see superclass. """
        return "agents"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """ Inherited, see superclass. """
        return AgentsFeature  # type: ignore

    def get_features_from_scenario(self, scenario: AbstractScenario) -> AgentsFeature:
        """ Inherited, see superclass. """
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

        return _compute_feature(sampled_past_ego_states, sampled_past_observations, time_stamps)

    def get_features_from_simulation(self, ego_states: List[EgoState], observations: List[Observation],
                                     meta_data: FeatureBuilderMetaData) -> AgentsFeature:
        """ Inherited, see superclass. """
        assert isinstance(observations[0], Detections), \
            f"Expected observation of type Detection, got {type(observations)}"

        interval_time = ego_states[1].time_point.time_s - ego_states[0].time_point.time_s

        present_observation = observations[-1]
        past_observations = observations[:-1]
        indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, interval_time)
        sampled_past_observations = [cast(Detections, past_observations[-idx]).boxes for idx in reversed(indices)]
        sampled_past_observations = sampled_past_observations + [cast(Detections, present_observation).boxes]

        present_ego_state = ego_states[-1]
        past_ego_states = ego_states[:-1]
        sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
        time_stamps = [state.time_point for state in sampled_past_ego_states]

        return _compute_feature(sampled_past_ego_states, sampled_past_observations, time_stamps)
