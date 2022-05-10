from datasets.perm_mnist import PermutedMNIST
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.rot_mnist import RotatedMNIST
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_core50 import SequentialCore50
from datasets.seq_cifar100_20 import SequentialCIFAR100_20
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.mnist_360 import MNIST360
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    PermutedMNIST.NAME: PermutedMNIST,
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialCIFAR100_20.NAME: SequentialCIFAR100_20,
    SequentialCore50.NAME: SequentialCore50,
    RotatedMNIST.NAME: RotatedMNIST,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    MNIST360.NAME: MNIST360
}

GCL_NAMES = {
    MNIST360.NAME: MNIST360
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
