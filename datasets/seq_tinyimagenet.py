import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize

import torch
from collections.abc import Sequence
import numbers
from torch import Tensor

class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyTinyImagenet(TinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img2 = self.transform(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, img2, target, not_aug_img, self.logits[index]

        return img, img2, target,  not_aug_img

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

# class GaussianBlur(torch.nn.Module):
#     """Blurs image with randomly chosen Gaussian blur.
#     If the image is torch Tensor, it is expected
#     to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
#     Args:
#         kernel_size (int or sequence): Size of the Gaussian kernel.
#         sigma (float or tuple of float (min, max)): Standard deviation to be used for
#             creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
#             of float (min, max), sigma is chosen uniformly at random to lie in the
#             given range.
#     Returns:
#         PIL Image or Tensor: Gaussian blurred version of the input image.
#     """

#     def __init__(self, kernel_size, sigma=(0.1, 2.0)):
#         super().__init__()
#         self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
#         for ks in self.kernel_size:
#             if ks <= 0 or ks % 2 == 0:
#                 raise ValueError("Kernel size value should be an odd and positive number.")

#         if isinstance(sigma, numbers.Number):
#             if sigma <= 0:
#                 raise ValueError("If sigma is a single number, it must be positive.")
#             sigma = (sigma, sigma)
#         elif isinstance(sigma, Sequence) and len(sigma) == 2:
#             if not 0.0 < sigma[0] <= sigma[1]:
#                 raise ValueError("sigma values should be positive and of the form (min, max).")
#         else:
#             raise ValueError("sigma should be a single number or a list/tuple with length 2.")

#         self.sigma = sigma

#     @staticmethod
#     def get_params(sigma_min: float, sigma_max: float) -> float:
#         """Choose sigma for random gaussian blurring.
#         Args:
#             sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
#             sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.
#         Returns:
#             float: Standard deviation to be passed to calculate kernel for gaussian blurring.
#         """
#         return torch.empty(1).uniform_(sigma_min, sigma_max).item()

#     def forward(self, img: Tensor) -> Tensor:
#         """
#         Args:
#             img (PIL Image or Tensor): image to be blurred.
#         Returns:
#             PIL Image or Tensor: Gaussian blurred image
#         """
#         sigma = self.get_params(self.sigma[0], self.sigma[1])
#         return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])

#     def __repr__(self):
#         s = "(kernel_size={}, ".format(self.kernel_size)
#         s += "sigma={})".format(self.sigma)
#         return self.__class__.__name__ + s

class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10

    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    TRANSFORM = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomApply([color_jitter], p = 0.7),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomGrayscale(p = 0.3),
                # transforms.RandomApply([GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821)),
            ])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                 train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                        train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_denormalization_transform()])

        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                            train=True, download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader


    @staticmethod
    def get_backbone():
        pass
        #return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK
        #                * SequentialTinyImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform
