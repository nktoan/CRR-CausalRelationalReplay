import os
import importlib
import torch
from utils.metrics import FocalLoss

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args, backbone, loss, transform):
    heavy_dataset = ['seq-cifar10']
    giant_dataset = ['seq-cifar100', 'seq-tinyimg', 'seq-cifar100_20', 'seq-core50']

    if (args.model[:3] == 'crr'):
        loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        #loss = FocalLoss(gamma = 2.0, alpha = 0.5)

        if (args.dataset in heavy_dataset):
            #loss = FocalLoss()
            backbone = 'resnet18'
        elif (args.dataset in giant_dataset):
            backbone = 'resnet18'
        else:
            backbone = 'mlp'
    
    return names[args.model](backbone, loss, args, transform)
