import torch.nn as nn
from torch.optim import Adam, SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.buffer import Buffer

from backbone.MyResNet import resnet18, resnet34, resnet50, resnet101
from backbone.MLP import get_mlp

backbone_model_dict = {
    'resnet18': [resnet18(), 512],
    'resnet34': [resnet34(), 512],
    'resnet50': [resnet50(), 2048],
    'resnet101': [resnet101(), 2048],
    'mlp' : [get_mlp(), 100],
}

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.scheduler = None
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

class ModifiedContinualModel(nn.Module):
    """
    Modified Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: str, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ModifiedContinualModel, self).__init__()

        self.net, self.dim_in = backbone_model_dict[backbone]
        self.loss = loss
        self.args = args
        self.transform = transform
        #self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.scheduler = None
        self.device = get_device()

    def forward_one(self, x : torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass of an input.
        Args:
            x: the input tensor.
        Return:
            the output tensor (feature vector) after going through the backbones. (feature extractors).
        """
        pass
    def get_buffer(self) -> Buffer:
        """
        Return the buffer used for this model
        """
        pass
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        pass

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
