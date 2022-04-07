# Useful imports
from pathlib import Path
import hydra

import logging

from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.model_builder import build_nn_model
from nuplan.planning.script.builders.training_builder import build_lightning_datamodule
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

import libpy_solver_util as py_solver_util

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Location of path with all training configs
CONFIG_PATH = '../nuplan/planning/script/config/training'
CONFIG_NAME = 'default_training'

# Create a temporary directory to store the cache and experiment artifacts
SAVE_DIR = Path('/home/lain/jiangpeng/nuplan/exp')  # optionally replace with persistent dir
EXPERIMENT = 'training_lmm_experiment'
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
    'data_loader.params.num_workers=1',
    '+data_loader.params.shuffle=True',
    #'resume_training=True',
    'worker=sequential',
])

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
    Dataloader = DataLoader(dataset=Dataset, **cfg.data_loader.params, collate_fn=FeatureCollate())

    print("Here is the data:")
    traj_res = []
    for id, data in enumerate(Dataloader):

        # the data to be used 
        # data[0]是vector_map, data[1]是Trajectory
        lanes = data[0]['vector_map'].coords[0][:, 0].numpy() # lanes (int * 2)
        connections = data[0]['vector_map'].multi_scale_connections[0][1].numpy() # connections (int * 2)
        ego = data[0]['agents'].ego[0].numpy()
        ego_pose = ego[:3] # (1*3) x, y, heading
        ego_size = ego[3:] # (1*2) width, length
        agents = data[0]['agents'].agents[0][-1].numpy()
        agent_pose = (agents[:,0:3]) # (num * 3) x, y, heading
        agents_velocity = (agents[:,3 : 5]) # (num * 2) x, y
        agents_size = (agents[:,6:]) # (num * 2)

        data_trans = tuple((lanes, connections, ego, agents))
        # print(data_trans)
        in_obj = py_solver_util.new_a_instance()
        result = py_solver_util.build_a_output(in_obj, data_trans)
        print("You succeed calling C++ using Python, here is your result (x, y, velocity, acceleration, curvature:  ):")
        for i in range(len(result)):
            print(result[i])
        break
        # visualize(y)


if __name__ == '__main__':
    main()