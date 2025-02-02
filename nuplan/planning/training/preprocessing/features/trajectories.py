from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.common.actor_state.utils import lazy_property
import numpy as np

@dataclass
class Trajectories(AbstractModelFeature):
    """
    A feature that contains multiple trajectories
    """
    trajectories: List[Trajectory]

    def to_feature_tensor(self) -> AbstractModelFeature:
        """ Implemented. See interface. """
        return Trajectories(trajectories=[trajectory.to_feature_tensor() for trajectory in self.trajectories])

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        """ Implemented. See interface. """
        return Trajectories(trajectories=[trajectory.to_device(device) for trajectory in self.trajectories])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        """ Implemented. See interface. """
        return Trajectories(trajectories=data["trajectories"])

    @property
    def number_of_trajectories(self) -> int:
        """
        :return: number of trajectories in this feature
        """
        return len(self.trajectories)
    
    @lazy_property
    def tensor(self) -> torch.Tensor:
        return self._tensor()

    def _tensor(self) -> torch.Tensor:
        return torch.stack([trajectory.data for trajectory in self.trajectories],dim=1)