from typing import List, cast

import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories

import numpy as np
from torch import Tensor

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
        self._fn = self.pytorch_neg_multi_log_likelihood_batch
    
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
    
    def pytorch_neg_multi_log_likelihood_batch(self, 
        gt: Tensor, pred: Tensor, confidences: Tensor
    ) -> Tensor:
        """
        Compute a negative log-likelihood for the multi-modal scenario.
        log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
        https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        https://leimao.github.io/blog/LogSumExp/
        Args:
            gt (Tensor): array of shape (bs)x(time)x(3D coords)
            pred (Tensor): array of shape (bs)x(modes)x(time)x(3D coords)
            confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
            avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
        Returns:
            Tensor: negative log-likelihood for this example, a single float number
        """
        assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
        batch_size, num_modes, future_len, num_coords = pred.shape

        assert gt.shape == (batch_size, future_len, num_coords), f"expected 3D (Time x Coords) array for gt, got {gt.shape}"
        assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
        assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
        # assert all data are valid
        assert torch.isfinite(pred).all(), "invalid value found in pred"
        assert torch.isfinite(gt).all(), "invalid value found in gt"
        assert torch.isfinite(confidences).all(), "invalid value found in confidences"

        # convert to (batch_size, num_modes, future_len, num_coords)
        gt = torch.unsqueeze(gt, 1)  # add modes

        # error (batch_size, num_modes, future_len)
        error = torch.sum((gt - pred) ** 2, dim=-1)  # reduce coords and use availability

        with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
            # error (batch_size, num_modes)
            error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

        # use max aggregator on modes for numerical stability
        # error (batch_size, num_modes)
        max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
        error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
        # print("error", error)
        return torch.mean(error)

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

        pred_tensor = (cast(Trajectories, predictions["trajectories"])).tensor()
        targ_tensor = cast(Trajectory, targets["trajectory"]).data
        
        confidence = predictions["confidence"]

        res = self.pytorch_neg_multi_log_likelihood_batch(targ_tensor, pred_tensor, confidence)
        return res