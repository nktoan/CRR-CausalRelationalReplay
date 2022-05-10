import torch
import torch.nn as nn

from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger

from argparse import Namespace
from typing import Tuple, Union
import sys
import numpy as np
import gc

from models.utils.continual_model import ContinualModel, ModifiedContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from datasets import get_dataset
from utils.buffer import Buffer

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    #[9, 5, 5]: 
    if (type(dataset.N_CLASSES_PER_TASK) == list):  
        outputs[:, 0: sum(dataset.N_CLASSES_PER_TASK[:k])] = -float('inf')
        outputs[:, sum(dataset.N_CLASSES_PER_TASK[:(k+1)]): sum(dataset.N_CLASSES_PER_TASK)] = -float('inf')
    else:
        outputs[:, 0: k * dataset.N_CLASSES_PER_TASK] = -float('inf')
        outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
                dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def mask_classes_max(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if (type(dataset.N_CLASSES_PER_TASK) == list):
        outputs[:, 0: sum(dataset.N_CLASSES_PER_TASK[:k])] = float('inf')
        outputs[:, sum(dataset.N_CLASSES_PER_TASK[:(k+1)]): sum(dataset.N_CLASSES_PER_TASK)] = float('inf')
    else:
        outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = float('inf')
        outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
                dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = float('inf')

def validate(model: Union[ModifiedContinualModel, ContinualModel], dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()
    accs, accs_mask_classes = [], []
    for k, val_loader in enumerate(dataset.val_loaders):
        if last and k < len(dataset.val_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                if (model.NAME[:3] == 'crr'):
                    outputs = model.forward_test(inputs)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

            del inputs, labels, outputs, pred
            gc.collect()
            torch.cuda.empty_cache()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train()
    return accs, accs_mask_classes

def validate_ncm(model: Union[ModifiedContinualModel, ContinualModel], dataset: ContinualDataset, buffer: Buffer, last: bool = False) -> float:
    """
    Evaluates the accuracy of the model for each past task using nearest-class-mean (ncm trick)
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()
    accs, accs_mask_classes = [], []
    #Find means of each class in the buffer
    #with torch.no_grad():
    #    model.compute_class_means()

    class_means = model.get_class_means()

    for k, val_loader in enumerate(dataset.val_loaders):
        if last and k < len(dataset.val_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(model.device)
            # if 'class-il' not in model.COMPATIBILITY: just use for cil-scenario (crr model)
            #     outputs = model(inputs, k)
            # else:
            features = model.forward_one(inputs)
            features = features.to('cpu')
            for j in range(features.size(0)):  # Normalize
                features.data[j] = features.data[j] / features.data[j].norm()
            features = features.unsqueeze(2)  # (batch_size, feature_size, 1)
            means = torch.stack([class_means[cls_] for cls_ in range(len(class_means))])  # (n_classes, feature_size)
            means = torch.stack([means] * inputs.size(0))  # (batch_size, n_classes, feature_size)
            means = means.transpose(1, 2)
            features = features.expand_as(means)  # (batch_size, feature_size, n_classes)
            dists = (features - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)

            _, predicted = dists.min(1)

            #_, pred = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes_max(dists, dataset, k)
                _, predicted = dists.min(1)
                correct_mask_classes += torch.sum(predicted == labels).item()

            del inputs, labels, features
            gc.collect()
            torch.cuda.empty_cache()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train()
    return accs, accs_mask_classes

def evaluate(model: Union[ModifiedContinualModel, ContinualModel], dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                if (model.NAME[:3] == 'crr'):
                    outputs = model.forward_test(inputs)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

            del inputs, labels, outputs, pred
            gc.collect()
            torch.cuda.empty_cache()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train()
    return accs, accs_mask_classes

def evaluate_ncm(model: Union[ModifiedContinualModel, ContinualModel], dataset: ContinualDataset, buffer: Buffer, last: bool = False) -> float:
    """
    Evaluates the accuracy of the model for each past task using nearest-class-mean (ncm trick)
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()
    accs, accs_mask_classes, accs_prediction, accs_prediction_mask_classes = [], [], [], []
    #Find means of each class in the buffer
    #with torch.no_grad():
    #    model.compute_class_means()

    class_means = model.get_class_means()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_classification, correct_classification_mask_classes, correct_mask_classes, total  = 0.0, 0.0, 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(model.device)

            with torch.no_grad():
                features = model.forward_one(inputs)
                prediction_score = model.forward_test(inputs)
            
            prediction_score = prediction_score.to('cpu')
            features = features.to('cpu')
            features = features.unsqueeze(2)  # (batch_size, feature_size, 1)

            means = torch.stack([class_means[cls_] for cls_ in range(len(class_means))])  # (n_classes, feature_size)
            means = torch.stack([means] * inputs.size(0))  # (batch_size, n_classes, feature_size)
            means = means.transpose(1, 2)

            features = features.expand_as(means)  # (batch_size, feature_size, n_classes)
            
            # l2 - distance or cosine distance.
            # 1. With mean
            
            dists = (features - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)

            # dists_ens = (dists - dists.min(1)[0].unsqueeze(1).expand_as(dists))/(dists.max(1)[0].unsqueeze(1).expand_as(dists) - dists.min(1)[0].unsqueeze(1).expand_as(dists))
            # dists_ens = torch.softmax(dists_ens, axis = 1)
            # dists_ens = 1 - dists_ens
            
            _, prediction_score_1 = dists.min(1)
            _, prediction_score_2 = prediction_score.max(1)

            correct += torch.sum(prediction_score_1 == labels).item()
            correct_classification += torch.sum(prediction_score_2 == labels).item()

            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes_max(dists, dataset, k)
                _, prediction_score_1 = dists.min(1)
                correct_mask_classes += torch.sum(prediction_score_1 == labels).item()

                mask_classes(prediction_score, dataset, k)
                _, prediction_score_2 = prediction_score.max(1)
                correct_classification_mask_classes += torch.sum(prediction_score_2 == labels).item()

            del inputs, labels, features
            gc.collect()
            torch.cuda.empty_cache()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

        accs_prediction.append(correct_classification / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_prediction_mask_classes.append(correct_classification_mask_classes / total * 100)

    model.train()

    # accs_after_task = pd.DataFrame()
    # n_tasks = len(accs)

    # for i in range (n_tasks):
    #     accs_after_task[f'acc_task_{i + 1}'] = accs[i]

    # accs_after_task.to_csv(f'./acc_task_{n_tasks}_{model.NAME}.csv', index = False)

    return accs, accs_mask_classes, accs_prediction, accs_prediction_mask_classes

def evaluate_prototypes(model: Union[ModifiedContinualModel, ContinualModel], dataset: ContinualDataset, buffer: Buffer, last: bool = False) -> float:
    """
    Evaluates the accuracy of the model for each past task using nearest-neighbor-prototypes (tricks)
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()
    accs, accs_mask_classes = [], []

    prototypes, prototypes_label = model.get_prototypes(300) # 20 prototypes per classes.
    
    prototypes = prototypes.to(model.device)
    features_prototypes = model.forward_one(prototypes)
    features_prototypes = features_prototypes.to('cpu')
    for j in range(features_prototypes.size(0)):  # Normalize
        features_prototypes.data[j] = features_prototypes.data[j] / features_prototypes.data[j].norm()

    features_prototypes = features_prototypes.detach().numpy()
    prototypes_label = prototypes_label.to('cpu')
    
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(features_prototypes, prototypes_label)

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(model.device)
            # if 'class-il' not in model.COMPATIBILITY: just use for cil-scenario (crr model)
            #     outputs = model(inputs, k)
            # else:
            features = model.forward_one(inputs)
            features = features.to('cpu')
            for j in range(features.size(0)):  # Normalize
                features.data[j] = features.data[j] / features.data[j].norm()

            features = features.detach().numpy()
            dists = neigh.predict_proba(features)
            dists = torch.tensor(dists)

            _, predicted = dists.max(1)

            #_, pred = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(dists, dataset, k)
                _, predicted = dists.max(1)
                correct_mask_classes += torch.sum(predicted == labels).item()

            del inputs, labels, features
            gc.collect()
            torch.cuda.empty_cache()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    del prototypes, prototypes_label, features_prototypes
    gc.collect()
    torch.cuda.empty_cache()   

    model.train()
    return accs, accs_mask_classes

def train(model: Union[ContinualDataset, ModifiedContinualModel], dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model = model.to(model.device)
    results, results_mask_classes = [], []

    model_stash = create_stash(model, args, dataset)

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.train()
        if (args.validation):
            _, _, _ = dataset_copy.get_data_loaders()
        else:
            _, _ = dataset_copy.get_data_loaders()

    if model.NAME != 'icarl' and model.NAME != 'pnn':
        if (model.NAME[:3] == 'crr'):
            buffer = model.get_buffer()
            random_results_class, random_results_task = None, None #evaluate_ncm(model, dataset_copy, buffer)
        else:
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    #scheduler = model.get_scheduler()

    for t in range(dataset.N_TASKS):
        model.train()
        if (args.validation):
            train_loader, val_loader, test_loader = dataset.get_data_loaders()
        else:
            train_loader, test_loader = dataset.get_data_loaders()

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            if (model.NAME[:3] == 'crr'):
                buffer = model.get_buffer()
                accs = evaluate_ncm(model, dataset, buffer, last = True)
            else:
                accs = evaluate(model, dataset, last = True)

            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        
        mx_val_score = -1
        cnt_early_stopping = 0

        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, inputs_2, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    inputs_2 = inputs_2.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, inputs_2, labels, not_aug_inputs, logits)

                    del logits
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    inputs, inputs_2, labels, not_aug_inputs = data
                    inputs, inputs_2, labels = inputs.to(model.device), inputs_2.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, inputs_2, labels, not_aug_inputs)

                if (t == 0):
                    progress_bar(i, len(train_loader), epoch, t, loss)
                elif (epoch == args.n_epochs - 1 or epoch == 0):
                    progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                model_stash['batch_idx'] = i + 1

                del inputs, labels
                gc.collect()
                torch.cuda.empty_cache()
            
            if (args.validation):
                # Evaluate the model
                if (t > 0):
                    if (model.NAME[:3] == 'crr'):
                        buffer = model.get_buffer()
                        accs_val = validate_ncm(model, dataset, buffer)
                    else:
                        accs_val = validate(model, dataset)

                    mean_acc_val = np.mean(accs_val, axis=1)

                    if (epoch % 8 == 0):
                        print('\nValidation Accuracy:')
                        print_mean_accuracy(mean_acc_val, t + 1, dataset.SETTING)

                    if (epoch >= 30):
                        cnt_early_stopping += 1

                    if (mean_acc_val[0] > mx_val_score): #Early stopping
                        mx_val_score = mean_acc_val[0]
                        cnt_early_stopping = 0
                    
                    if (cnt_early_stopping >= 10 and epoch >= 30):
                        for epoch_ in range(epoch, args.n_epochs):
                            model.step_scheduler(args.n_epochs * t + (epoch_ + 1))

                        break
            
            scheduler = model.get_scheduler()
            if (scheduler is not None):
                lr_before = scheduler.get_lr()
                model.step_scheduler(args.n_epochs * t + (epoch + 1))
                scheduler = model.get_scheduler()
                if (lr_before != scheduler.get_lr()):
                    print(f'\nLearning rate of model decrease to {scheduler.get_lr()} at task: {t + 1}, epoch: {epoch + 1}')

            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0

            # if ((epoch+1) % 10 == 0):
                # torch.save(model.state_dict(), f'None/cifar10_buffer500_model_{epoch+1}e_task_{t+1}.pt')

        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            # model.step_scheduler(-1)
            model.end_task(dataset)

        if (model.NAME[:3] == 'crr'):
            buffer = model.get_buffer()
            accs = evaluate_ncm(model, dataset, buffer)
        else:
            accs = evaluate(model, dataset)

        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print('\nTest Accuracy:')
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn' and model.NAME[:3] != 'crr':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
