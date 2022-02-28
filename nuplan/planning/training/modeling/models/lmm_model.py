from typing import List

from timm.models.layers.conv2d_same import Conv2dSame
import timm
from torch import nn
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.nn_model import NNModule
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())

def convert_predictions_to_trajectories(predictions: torch.Tensor, num_head: int) -> Trajectories:
    """
    Convert predictions tensor to Trajectories
    :param predictions: tensor from network, [batch_size, num_head*num_poses*state_size]
    :return: Trajectories
    """
    num_batches = predictions.shape[0]
    data = (predictions.view(num_batches, num_head, -1, Trajectory.state_size())).permute(1,0,2,3)
    trajectories: Trajectories = Trajectories([Trajectory(data=data[i,:,:,:]) for i in range(num_head)])

    return trajectories


class LMMModel(NNModule):

    def __init__(self,
                 feature_builders: List[AbstractFeatureBuilder],
                 target_builders: List[AbstractTargetBuilder],
                 model_name: str,
                 pretrained: bool,
                 num_input_channels: int,
                 num_features_per_pose: int,
                 future_trajectory_sampling: TrajectorySampling,
                 num_head: int
                 ):
        """
        Wrapper around raster-based CNN model
        :param feature_builders: list of builders for features
        :param target_builders: list of builders for targets
        :param model_name: name of the model (e.g. resnet_50, efficientnet_b3)
        :param pretrained: whether the model will be pretrained
        :param num_input_channels: number of input channel of the raster model.
        :param num_features_per_pose: number of features per single pose
        :param future_trajectory_sampling: parameters of predicted trajectory
        """
        super().__init__(feature_builders=feature_builders, target_builders=target_builders,
                         future_trajectory_sampling=future_trajectory_sampling)
        
        self.num_head = num_head
        num_output_features = future_trajectory_sampling.num_poses * num_features_per_pose* num_head
        self._model = timm.create_model(model_name, pretrained=pretrained)
        self._model.conv_stem = Conv2dSame(
            num_input_channels,
            self._model.conv_stem.out_channels,
            kernel_size=self._model.conv_stem.kernel_size,
            stride=self._model.conv_stem.stride,
            padding=self._model.conv_stem.padding,
            bias=False,
        )
        self.backbone_out_features = self._model.classifier.in_features
        self._model.classifier = nn.Sequential(
            nn.Identity(),
            nn.Linear(
                in_features=self._model.classifier.in_features,
                out_features=self.backbone_out_features,
            ),
        )

        self.lin_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                in_features=self.backbone_out_features,
                out_features=num_output_features,
            ),
        )
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "raster": Raster,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        raster: Raster = features["raster"]

        predictions = self.lin_head(self._model.forward(raster.data))

        data: Trajectories = convert_predictions_to_trajectories(predictions, self.num_head)
        return {"trajectories": data}