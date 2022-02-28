# Useful imports
import os
from pathlib import Path
import tempfile

import hydra

# Location of path with all training configs
CONFIG_PATH = '../nuplan/planning/script/config/training'
CONFIG_NAME = 'default_training'

# Create a temporary directory to store the cache and experiment artifacts
SAVE_DIR = Path('/home/fla/temp') / 'tutorial_nuplan_framework'  # optionally replace with persistent dir
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
    '+training=training_lmm_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=500',  # Choose 500 scenarios to train with
    'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.01',  # subsample scenarios from 20Hz to 0.2Hz
    'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
    'lightning.trainer.params.max_epochs=100',
    'data_loader.params.batch_size=16',
    'data_loader.params.num_workers=8',
    'worker=sequential'
])

from nuplan.planning.script.run_training import main as main_train

# Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
main_train(cfg)