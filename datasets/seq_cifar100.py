# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
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
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    """
    args = main.parse_args()
    if args.optim_upscale == 1:
        if backbone == 'resnet18' or backbone == 'vit_b_16' or backbone == 'vit_b_32':
            image_resize = 256
        else:
            image_resize = 232

        image_crop       = 224

        TRANSFORM = transforms.Compose(
                                      [transforms.Resize(image_resize, interpolation=transforms.InterpolationMode.BILINEAR),
                                       transforms.RandomCrop(image_crop),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        TEST_TRANSFORM = transforms.Compose(
                                      [transforms.Resize(image_resize, interpolation=transforms.InterpolationMode.BILINEAR),
                                       transforms.CentralCrop(image_crop),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])    
    else:
        TRANSFORM = transforms.Compose(
                              [transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
      
        TEST_TRANSFORM = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    """
    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR10', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = self.TEST_TRANSFORM

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    ### START --- aghinea
    def parse_args():
        parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    
        #To use this argument add the same in utils/args.py --> add_management_args
        parser.add_argument('--optim_upscale',type=int,help='Upscale images to default size of pre-trained model recipe. 0 (no upscale), 1 (upscale)',default=0,choices=[0,1])
        parser.add_argument('--backbone',type=str,help='Pre-trained backbone to use, choose from pytorch models: resnet18, resnet34, resnet50, resnet101, resnet152, vit_b_16, vit_b_32', default='resnet18')
        
        args = parser.parse_args()
        return args  
  ### END   --- aghinea
  
    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

