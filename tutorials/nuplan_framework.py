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
EXPERIMENT = 'training_raster_experiment'
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
    '+training=training_raster_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=500',  # Choose 500 scenarios to train with
    'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.01',  # subsample scenarios from 20Hz to 0.2Hz
    'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
    'lightning.trainer.params.max_epochs=10',
    'data_loader.params.batch_size=1',
    'data_loader.params.num_workers=0'
])

from nuplan.planning.script.run_training import main as main_train

# Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
main_train(cfg)

# # Location of path with all simulation configs
# CONFIG_PATH = '../nuplan/planning/script/config/simulation'
# CONFIG_NAME = 'default_simulation'

# # Select the planner and simulation challenge
# PLANNER = 'simple_planner'  # [simple_planner, ml_planner]
# CHALLENGE = 'challenge_1_open_loop_boxes'  # [challenge_1_open_loop_boxes, challenge_3_closed_loop_nonreactive_agents, challenge_4_closed_loop_reactive_agents]
# DATASET_PARAMS = [
#     'scenario_builder=nuplan_mini',  # use nuplan mini database
#     'scenario_builder/nuplan/scenario_filter=all_scenarios',  # initially select all scenarios in the database
#     'scenario_builder.nuplan.scenario_filter.scenario_types=[nearby_dense_vehicle_traffic, ego_at_pudo, ego_starts_unprotected_cross_turn, ego_high_curvature]',  # select scenario types
#     'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=10',  # use 10 scenarios per scenario type
#     'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.05',  # subsample 20s scenario from 20Hz to 1Hz
# ]

# # Name of the experiment
# EXPERIMENT = 'simulation_simple_experiment'

# # Initialize configuration management system
# hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
# hydra.initialize(config_path=CONFIG_PATH)

# # Compose the configuration
# cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
#     f'experiment_name={EXPERIMENT}',
#     f'group={SAVE_DIR}',
#     f'planner={PLANNER}',
#     f'+simulation={CHALLENGE}',
#     *DATASET_PARAMS,
# ])

# from nuplan.planning.script.run_simulation import main as main_simulation

# # Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
# main_simulation(cfg)

# # Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
# parent_dir = Path(SAVE_DIR) / EXPERIMENT
# results_dir = list(parent_dir.iterdir())[0]  # get the child dir
# nuboard_file_1 = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]

# # Location of path with all simulation configs
# CONFIG_PATH = '../nuplan/planning/script/config/simulation'
# CONFIG_NAME = 'default_simulation'

# # Get the checkpoint of the trained model
# last_experiment = sorted(os.listdir(LOG_DIR))[-1]
# train_experiment_dir = sorted(Path(LOG_DIR).iterdir())[-1]
# checkpoint = sorted((train_experiment_dir / 'checkpoints').iterdir())[-1]

# MODEL_PATH = str(checkpoint).replace("=", "\=")

# # Name of the experiment
# EXPERIMENT = 'simulation_raster_experiment'

# # Initialize configuration management system
# hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
# hydra.initialize(config_path=CONFIG_PATH)

# # Compose the configuration
# cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
#     f'experiment_name={EXPERIMENT}',
#     f'group={SAVE_DIR}',
#     'planner=ml_planner',
#     'model=raster_model',
#     'planner.model_config=${model}',  # hydra notation to select model config
#     f'planner.checkpoint_path={MODEL_PATH}',  # this path can be replaced by the checkpoint of the model trained in the previous section
#     f'+simulation={CHALLENGE}',
#     *DATASET_PARAMS,
# ])

# # Run the simulation loop
# main_simulation(cfg)

# # Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
# parent_dir = Path(SAVE_DIR) / EXPERIMENT
# results_dir = list(parent_dir.iterdir())[0]  # get the child dir
# nuboard_file_2 = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]

# # Location of path with all nuBoard configs
# CONFIG_PATH = '../nuplan/planning/script/config/nuboard'
# CONFIG_NAME = 'default_nuboard'

# # Initialize configuration management system
# hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
# hydra.initialize(config_path=CONFIG_PATH)

# # Compose the configuration
# cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
#     'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
#     f'simulation_path={[nuboard_file_1, nuboard_file_2]}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
# ])

# from nuplan.planning.script.run_nuboard import main as main_nuboard

# # Run nuBoard
# main_nuboard(cfg)