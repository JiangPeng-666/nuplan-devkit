from typing import List, cast

import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories


class MultiImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'multi_imitation_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='mean')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='mean')
    
    def multi_average_displacement_error(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectories: Trajectories = predictions["trajectories"]
        targets_trajectory: Trajectory = targets["trajectory"]
        return torch.norm(torch.tensor([torch.norm(predicted_trajectories.trajectories[i].xy - targets_trajectory.xy, dim=-1).mean() for i in range(predicted_trajectories.number_of_trajectories)])).mean().cuda().requires_grad_(True)

    def multi_average_heading_error(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """

        predicted_trajectories: Trajectories = predictions["trajectories"]
        targets_trajectory: Trajectory = targets["trajectory"]
        def get_error_single(predicted_trajectories, targets_trajectory, i):
            error = torch.abs(predicted_trajectories.trajectories[i].heading - targets_trajectory.heading)
            return torch.atan2(torch.sin(error), torch.cos(error)).mean()

        return torch.norm(torch.tensor([get_error_single(predicted_trajectories, targets_trajectory, i) for i in range(predicted_trajectories.number_of_trajectories)])).mean().cuda().requires_grad_(True)

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """

        predicted_trajectories = cast(Trajectories, predictions["trajectories"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])
        
        res = self._weight * (
                self._fn_xy(predicted_trajectories.trajectories[0].xy, targets_trajectory.xy) +
                self._fn_heading(predicted_trajectories.trajectories[0].heading, targets_trajectory.heading))
        for i in range(predicted_trajectories.number_of_trajectories-1):
            res += self._weight * (
                self._fn_xy(predicted_trajectories.trajectories[i+1].xy, targets_trajectory.xy) +
                self._fn_heading(predicted_trajectories.trajectories[i+1].heading, targets_trajectory.heading))
        return res/3
