import torch
from datasets import get_gcl_dataset
from models import get_model
from utils.status import progress_bar
from utils.tb_logger import *
from utils.status import create_fake_stash
from models.utils.continual_model import ContinualModel, ModifiedContinualModel
from argparse import Namespace
from utils.buffer import Buffer
import gc


def evaluate(model: ContinualModel, dataset) -> float:
    """
    Evaluates the final accuracy of the model.
    :param model: the model to be evaluated
    :param dataset: the GCL dataset at hand
    :return: a float value that indicates the accuracy
    """
    model.eval()
    correct, total = 0, 0
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        if (model.NAME == 'crr' or model.NAME == 'crrpp'):
            outputs = model.forward_test(inputs)
        else:
            outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        correct += torch.sum(predicted == labels).item()
        total += labels.shape[0]

        del inputs, labels
        gc.collect()
        torch.cuda.empty_cache()

    acc = correct / total * 100
    return acc

def evaluate_ncm(model: ModifiedContinualModel, dataset, buffer: Buffer) -> float:
    """
    Evaluates the final accuracy of the model using Nearest-Class-Mean.
    :param model: the model to be evaluated
    :param dataset: the GCL dataset at hand
    :return: a float value that indicates the accuracy
    """
    model.eval()
    correct, total = 0, 0

    #Find means of each class in the buffer
    # with torch.no_grad():
    #     model.compute_class_means()

    class_means = model.get_class_means()

    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        features = model.forward_one(inputs)
        for j in range(features.size(0)):  # Normalize
            features.data[j] = features.data[j] / features.data[j].norm()
        features = features.unsqueeze(2)  # (batch_size, feature_size, 1)
        means = torch.stack([class_means[cls_] for cls_ in range(len(class_means))])  # (n_classes, feature_size)
        means = torch.stack([means] * inputs.size(0))  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)
        features = features.expand_as(means)  # (batch_size, feature_size, n_classes)
        dists = (features - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
        _, predicted = dists.min(1)

        #_, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted == labels).item()
        total += labels.shape[0]

        del inputs, labels
        gc.collect()
        torch.cuda.empty_cache()

    acc = correct / total * 100
    return acc


def train(args: Namespace):
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    if args.csv_log:
        from utils.loggers import CsvLogger

    dataset = get_gcl_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model = model.to(model.device)

    model_stash = create_fake_stash(model, args)

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)

    model.train()
    epoch, i = 0, 0
    while not dataset.train_over:
        inputs, labels, not_aug_inputs = dataset.get_train_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)
        loss = model.observe(inputs, labels, not_aug_inputs)
        progress_bar(i, dataset.LENGTH // args.batch_size, epoch, 'C', loss)
        if args.tensorboard:
            tb_logger.log_loss_gcl(loss, i)
        i += 1

        del inputs, labels
        gc.collect()
        torch.cuda.empty_cache()

    if model.NAME == 'joint_gcl' or model.NAME == 'crr':
      model.end_task(dataset)

    if (model.NAME == 'crr' or model.NAME == 'crrpp'):
        buffer = model.get_buffer()
        acc = evaluate_ncm(model, dataset, buffer)
    else:
        acc = evaluate(model, dataset)
    
    print('Accuracy:', acc)

    if args.csv_log:
        csv_logger.log(acc)
        csv_logger.write(vars(args))
