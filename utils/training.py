# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None
    
###START --- aghinea
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
###END   --- aghinea

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

@static_vars(all_labels=[],all_preds=[])
def evaluate(model: ContinualModel, dataset: ContinualDataset, args, last=False, current_task=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                
                if current_task == dataset.N_TASKS-1:
                    evaluate.all_preds.extend(pred.cpu()) 
                    evaluate.all_labels.extend(labels.cpu())
                
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        project = args.wandb_project

        if args.dataset == 'seq-tinyimg-hd':
            project = 'continual_tinyimagenethd'
            
        if args.upscale == 1:
            if args.dataset == 'seq-cifar10':
                project = 'continual_cifar10_upsampled'
            elif args.dataset == 'seq-cifar100':
                project = 'continual_cifar100_upsampled'
            elif args.dataset == 'seq-tinyimg':
                project = 'continual_tinyimagenet_upsampled'
            elif args.dataset == 'seq-imagenetR':
                project = 'continual_imagenetR'
        else:
            project = 'continual_benchmarks'
    
        wandb.init(dir='/home/aghinea/tmp/', project=project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()
        
    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy, args)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, args, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        ###START --- aghinea
        if args.plot_curve:
            accs = evaluate(model, dataset, args, last=False, current_task=t)
        else:
            accs = evaluate(model, dataset, args)
        ###END   --- aghinea
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)
    
    if args.plot_curve:
        if args.dataset == 'seq-cifar10':
            classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        if args.dataset == 'seq-cifar100':
            classes = ['beaver',
                         'dolphin',
                         'otter', 
                         'seal',
                         'whale',
                         'aquarium fish', 
                         'flatfish', 
                         'ray', 
                         'shark', 
                         'trout',
                         'orchids', 
                         'poppy', 
                         'rose', 
                         'sunflower', 
                         'tulip',
                         'bottle', 
                         'bowl', 
                         'can', 
                         'cup', 
                         'plate',
                         'apple', 
                         'mushroom', 
                         'orange', 
                         'pear', 
                         'sweet_pepper',
                         'clock', 
                         'computer keyboard', 
                         'lamp',
                         'telephone', 
                         'television',
                         'bed', 
                         'chair', 
                         'couch', 
                         'table', 
                         'wardrobe',
                         'bee', 
                         'beetle', 
                         'butterfly', 
                         'caterpillar', 
                         'cockroach',
                         'bear', 
                         'leopard', 
                         'lion', 
                         'tiger', 
                         'wolf',
                         'bridge', 
                         'castle', 
                         'house', 
                         'road', 
                         'skyscraper'
                         'cloud', 
                         'forest', 
                         'mountain', 
                         'plain',
                         'sea',
                         'camel', 
                         'cattle', 
                         'chimpanzee',
                         'elephant', 
                         'kangaroo'
                         'fox', 
                         'porcupine', 
                         'possum', 
                         'raccoon', 
                         'skunk',
                         'crab', 
                         'lobster', 
                         'snail', 
                         'spider', 
                         'worm',
                         'baby', 
                         'boy', 
                         'girl', 
                         'man', 
                         'woman',
                         'crocodile', 
                         'dinosaur', 
                         'lizard', 
                         'snake', 
                         'turtle',
                         'hamster', 
                         'mouse', 
                         'rabbit', 
                         'shrew',
                         'squirrel',
                         'maple_tree', 
                         'oak_tree', 
                         'palm_tree', 
                         'pine_tree', 
                         'willow_tree',
                         'bicycle', 
                         'bus', 
                         'motorcycle', 
                         'pickup _truck', 
                         'train'
                         'lawn-mower', 
                         'rocket', 
                         'streetcar', 
                         'tank', 
                         'tractor']
            
        wandb.log({'conf_matrix': wandb.sklearn.plot_confusion_matrix(evaluate.all_labels, evaluate.all_preds, classes)})
    
    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:            
        wandb.finish()
