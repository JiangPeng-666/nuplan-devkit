from typing import List

import torch
from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class AverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        return torch.norm(predicted_trajectory.xy - targets_trajectory.xy, dim=-1).mean()


class FinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        return torch.norm(predicted_trajectory.terminal_position - targets_trajectory.terminal_position, dim=-1).mean()


class AverageHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        errors = torch.abs(predicted_trajectory.heading - targets_trajectory.heading)
        return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()


class FinalHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectory: Trajectory = predictions["trajectory"]
        targets_trajectory: Trajectory = targets["trajectory"]

        errors = torch.abs(predicted_trajectory.terminal_heading - targets_trajectory.terminal_heading)
        return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()

class MultiAverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses for each head of the trajectory.
    """

    def __init__(self, name: str = 'multi_avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectories: Trajectories = predictions["trajectories"]
        targets_trajectory: Trajectory = targets["trajectory"]
        return torch.norm(torch.tensor([torch.norm(predicted_trajectories.trajectories[i].xy - targets_trajectory.xy, dim=-1).mean() for i in range(predicted_trajectories.number_of_trajectories)])).mean()
        # return torch.tensor([torch.norm(predicted_trajectories[i].xy - targets_trajectory.xy, dim=-1).mean() for i in predicted_trajectories.number_of_trajectories])


class MultiFinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectories: Trajectories = predictions["trajectories"]
        targets_trajectory: Trajectory = targets["trajectory"]

        return torch.norm(torch.tensor([torch.norm(predicted_trajectories.trajectories[i].terminal_position - targets_trajectory.terminal_position, dim=-1).mean() for i in range(predicted_trajectories.number_of_trajectories)])).mean()
        # return torch.tensor([torch.norm(predicted_trajectories[i].terminal_position - targets_trajectory.terminal_position, dim=-1).mean() for i in predicted_trajectories.number_of_trajectories])



class MultiAverageHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
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

        return torch.norm(torch.tensor([get_error_single(predicted_trajectories, targets_trajectory, i) for i in range(predicted_trajectories.number_of_trajectories)])).mean()
        # return torch.tensor([torch.atan2(torch.sin(torch.abs(predicted_trajectories[i].heading - targets_trajectory.heading)), torch.cos(torch.abs(predicted_trajectories[i].heading - targets_trajectory.heading))).mean() for i in predicted_trajectories.number_of_trajectories])


class MultiFinalHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """

        predicted_trajectories: Trajectories = predictions["trajectories"]
        targets_trajectory: Trajectory = targets["trajectory"]

        def get_error_single(predicted_trajectories, targets_trajectory, i):
            error = torch.abs(predicted_trajectories.trajectories[i].terminal_heading - targets_trajectory.terminal_heading)
            return torch.atan2(torch.sin(error), torch.cos(error)).mean()
        
        return torch.norm(torch.tensor([get_error_single(predicted_trajectories, targets_trajectory, i) for i in range(predicted_trajectories.number_of_trajectories)])).mean()
        # return torch.tensor([torch.atan2(torch.sin(torch.abs(predicted_trajectories[i].terminal_heading - targets_trajectory.terminal_heading)), torch.cos(torch.abs(predicted_trajectories[i].terminal_heading - targets_trajectory.terminal_heading))).mean() for i in predicted_trajectories.number_of_trajectories])