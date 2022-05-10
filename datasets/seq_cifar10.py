import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders, store_masked_loaders_val
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize

from datasets.transforms.cutout import Cutout
from datasets.transforms.autoaugment_extra import CIFAR10Policy

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img2 = self.transform(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, img2, target, not_aug_img, self.logits[index]

        return img, img2, target, not_aug_img

    def append_items(self, images, labels):
        """
        Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """
        if (images.ndim == 1):
            images = np.reshape(images,(1, -1))
        self.data = np.concatenate((self.data, images), axis=0) #images: shape[N, img]
        self.targets.extend([labels])

class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    TRANSFORM = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomApply([color_jitter], p = 0.7),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomGrayscale(p = 0.25),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615))
            ])

    WEAK_TRANSFORM = transforms.Compose(
        [
            transforms.CenterCrop(12),
            transforms.RandomAffine(degrees=(30, 70), translate=(0.5, 0.5), scale=(0.5, 0.75)),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615))
        ])

    BOTH_TRANSFORM = transforms.Compose(
        [
            transforms.CenterCrop(20),
            transforms.RandomApply([color_jitter], p = 0.6),
            transforms.RandomAffine(degrees=(30, 70), translate=(0.4, 0.4), scale=(0.5, 0.75)),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomGrayscale(p = 0.5),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615))
        ])
    
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        
        if (self.args.validation):
            train_dataset, val_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)

        test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                                    download=True, transform=test_transform)
        
        if (self.args.validation):
            train, val, test = store_masked_loaders_val(train_dataset, val_dataset, test_dataset, self)
            return train, val, test
        else:
            train, test = store_masked_loaders(train_dataset, test_dataset, self)
            return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        pass
        #return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
        #                * SequentialCIFAR10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform