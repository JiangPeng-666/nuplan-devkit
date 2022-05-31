# Useful imports
from pathlib import Path
import cv2
import hydra

import logging

from nuplan.common.actor_state.state_representation import StateSE2

from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.model_builder import build_nn_model
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

from nuplan.planning.training.preprocessing.feature_builders.lmm_feature_builder import LMMFeatureBuilder
from nuplan.planning.training.visualization.raster_visualization import Color, get_raster_with_trajectories_as_rgb_from_rgb
from nuplan.planning.training.visualization.raster_visualization import get_raster_with_trajectories_as_rgb
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from torch.utils.data import DataLoader
import numpy as np
import torch
import datetime
from matplotlib import pyplot as plt

import libpy_solver_util as py_solver_util

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Location of path with all training configs
CONFIG_PATH = '../nuplan/planning/script/config/training'
CONFIG_NAME = 'default_training'

# Create a temporary directory to store the cache and experiment artifacts
SAVE_DIR = Path('/home/lain/jiangpeng/nuplan/exp')  # optionally replace with persistent dir
EXPERIMENT = 'training_vector_experiment'
LOG_DIR = str(SAVE_DIR / EXPERIMENT)

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'group={str(SAVE_DIR)}',
    f'cache_dir={str(SAVE_DIR)}/cache',
    f'experiment_name={EXPERIMENT}',
    'py_func=train',
    '+training=training_vector_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=1000',  # MAX = 15890932 (v0.2)
    'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.01',  # subsample scenarios from 20Hz to 1Hz
    'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
    'lightning.trainer.params.max_epochs=100',
    'data_loader.params.batch_size=1',
    'data_loader.params.num_workers=0',
    '+data_loader.params.shuffle=False',
    #'resume_training=True',
    'worker=sequential',
])

def get_trajectory_from_output(input: np.ndarray) -> Trajectory:
    return Trajectory(torch.tensor(input))


def get_visual_image(pixel_size: float, scenario: AbstractScenario,
                        target_trajectory: Trajectory, predicted_trajectory: Trajectory,
                        feature_builder: LMMFeatureBuilder) -> np.ndarray:
    feature_raster = feature_builder.get_features_from_scenario(scenario).data # Raster, 
    image = get_raster_with_trajectories_as_rgb_from_rgb(
        pixel_size = pixel_size,
        raster = feature_raster,
        target_trajectory = target_trajectory, # 另一模型输出
        predicted_trajectory = predicted_trajectory, # epsilon输出
        target_trajectory_color= (140,0,0),
        predicted_trajectory_color = (183,135,10)
    )
    return image

def main():
    worker = build_worker(cfg)
    update_config_for_training(cfg)
    scenario_builder = build_scenario_builder(cfg)

    nn_model = build_nn_model(cfg.model)
    feature_builders = nn_model.get_list_of_required_feature()
    target_builder = nn_model.get_list_of_computed_target()
    computator = FeatureCachingPreprocessor(
        cache_dir=cfg.cache_dir,
        force_feature_computation=cfg.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builder,
    )        

    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    logger.info("Extracting all scenarios...")
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
    logger.info("Extracting all scenarios...DONE")

    Dataset = ScenarioDataset(scenarios=scenarios, feature_caching_preprocessor=computator)
    # Dataloader = DataLoader(dataset=Dataset, **cfg.data_loader.params, collate_fn=FeatureCollate())


    print('Dataset size:', len(Dataset))
    for id in range(len(Dataset)):
    # for id, data in enumerate(Dataloader):

        # data[0] is a dict of features, including vector_map and agents
        # data[1] is Trajectory output of the model
        # data[2] is the current scenario get from the dataset
        data = Dataset[id]

        # the data to be used 
        # the lanes are built directly from the scenerio
        lanes_epsi_list = VectorMapFeatureBuilder.get_features_from_scenario_for_epsilon(scenario=data[2])

        lanes_id, lanes_length, points = lanes_epsi_list[0], lanes_epsi_list[1], lanes_epsi_list[2]
        pre_connections, nxt_connections = lanes_epsi_list[3], lanes_epsi_list[4]
        left_connection, right_connection = lanes_epsi_list[5], lanes_epsi_list[6]

        agents_data = data[0]['agents'].serialize()

        ego_pose = agents_data['ego'][0].numpy().tolist() # (1*5) x, y, heading, acceleration, velocity
        ego_size = [2.297, 5.176] # (1*2) width, length, /m, from lmm_model.yaml
        ego_pose.extend(ego_size)
        ego = np.array(ego_pose) # (1 * 7)

        # in the dataloader, agents and obstacles are put together
        # distinguish them by their sizes and velocity here
        # nums * 8, pose((center)x, y, heading), velocity(x, y, yaw), width, length 
        agents = agents_data['agents'][0][0].numpy().tolist() 
        vehicles_list, obstacles_list = [], []
        for i in range(len(agents)):
            # if (agents[i][6] < 1 or agents[i][7] < 4) and (pow(agents[i][3], 2) + pow(agents[i][4], 2) < 1):
            if agents[i][6] < 1 or agents[i][7] < 4:
                obstacles_list.append(agents[i])
            else:
                vehicles_list.append(agents[i])

        # get the points of the obstacles as polygon
        for i in range(len(obstacles_list)):
            del obstacles_list[i][2:6]
            cur = obstacles_list[i]
            obstacles_list[i] = [cur[0] - cur[2]/2, cur[1] - cur[3]/2, cur[0] - cur[2]/2, cur[1] + cur[3]/2,
                                cur[0] + cur[2]/2, cur[1] - cur[3]/2, cur[0] + cur[2]/2, cur[1] + cur[3]/2]
            
        if len(vehicles_list) == 0:
            vehicles_list = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
        if len(obstacles_list) == 0:
            obstacles_list = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
        vehicles, obstacles = np.array(vehicles_list), np.array(obstacles_list)

        data_trans = tuple((lanes_id, lanes_length, points, pre_connections, nxt_connections,
            left_connection, right_connection, ego, vehicles, obstacles))

        # Visualize
        trajectory_sampling = TrajectorySampling(num_poses = 10, time_horizon = 1.5)
        lmm_feature_builder = LMMFeatureBuilder(
            map_features = {
                'LANE':255,
                'INTERSECTION' : 255,
                'STOP_LINE' : 128,
                'CROSSWALK' : 128
            },
            target_width = 224,
            target_height = 224,
            target_pixel_size = 0.5,
            ego_width = 2.297,
            ego_front_length =  4.049,
            ego_rear_length = 1.127,
            ego_longitudinal_offset = 0.0,
            baseline_path_thickness = 1,
            trajectory_sampling = trajectory_sampling,
            use_trafficlight = False,
            use_rgb = True,
        )

        # print("Finish preparing the {}-th data, passing the data to epsilon".format(id))

        # for debugging
        for i in range(len(vehicles_list)):
            plt.plot(vehicles_list[i][0], vehicles_list[i][1], marker = "o", markersize=10, color='#0000ff')
            plt.annotate("%s" %(i + 1), (vehicles_list[i][0], vehicles_list[i][1]), 
                    xytext=(0, 0), textcoords='offset points', weight='heavy')
        
        plt.plot(ego[0], ego[1], marker = "o",markersize=10, color='#ff0000')
        
        plt.show()
        now_time = datetime.datetime.now()
        now_time_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')
        plt_path = "mat_plt_{}.png".format(now_time_str)
        plt.savefig(plt_path)

        in_obj = py_solver_util.new_a_instance()
        # x, y, angle, velocity, acceleration
        result = np.array(py_solver_util.build_a_output(in_obj, data_trans)) # nums * 5

        if (len(result) > 1):
            epsi_out = np.delete(result, [3, 4], axis=1)
            epsi_out_state = []
            print("Succeed calling C++ using Python")
            flag = True
            for i in range(len(epsi_out)):
                epsi_out_state.append(StateSE2.deserialize(epsi_out[i]))

            epsi_out_relative = convert_absolute_to_relative_poses(StateSE2.deserialize(ego_pose[:3]), epsi_out_state)
            predicted_trajectory = get_trajectory_from_output(epsi_out_relative)
            image = get_visual_image(
                pixel_size = 0.5,
                scenario = data[2],
                target_trajectory= data[1]['trajectory'],
                predicted_trajectory= predicted_trajectory,
                feature_builder = lmm_feature_builder
            )
        else:
            print("Fail solving the scenario using epsilon.")
            flag = False
            image = get_visual_image(
                pixel_size = 0.5,
                scenario = data[2],
                target_trajectory= data[1]['trajectory'],
                predicted_trajectory= data[1]['trajectory'],
                feature_builder = lmm_feature_builder
            )


        now_time = datetime.datetime.now()
        now_time_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')
        if flag:
            res_path = "Succ_test_{}.png".format(now_time_str)
        else:
            res_path = "Fail_test_{}.png".format(now_time_str)
        cv2.imwrite(res_path, image)

        print("Preparing for the next data...")

        # break



if __name__ == '__main__':
    main()