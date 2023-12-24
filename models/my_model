# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    
    return parser

class MyModel(ContinualModel):
    NAME = 'my_model'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(MyModel, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs):

        
