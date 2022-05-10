import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

import torchvision

from argparse import Namespace
import time
import numpy as np
from copy import deepcopy

from utils.buffer import Buffer
from utils.args import *
from utils.mmd_critic import VectorEmbeddingDataset, select_criticisms, select_prototypes
from datasets import get_dataset

from backbone.MyResNet import resnet18, resnet34, resnet50, resnet101
from backbone.MLP import get_mlp

from models.utils.continual_model import ContinualModel, ModifiedContinualModel
from typing import Tuple
import gc

from sklearn.cluster import KMeans

backbone_model_dict = {
    'resnet18': [resnet18(), 512],
    'resnet34': [resnet34(), 512],
    'resnet50': [resnet50(), 2048],
    'resnet101': [resnet101(), 2048],
    'mlp' : [get_mlp(), 100],
}

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'Relational Reasoning.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default = 1.2,
                        help='Penalty weight for old pairs replay.')
    parser.add_argument('--beta', type=float, default = 1.5,
                        help='Penalty weight for old pairs distillation.')
    parser.add_argument('--gamma', type=float, default = 0.002,
                        help='Hyperparameters for MMD-critic.')
    parser.add_argument('--temp', type=float, default = 2.5,
                        help='Temperature for distillation logits reasoning features.')
    parser.add_argument('--buffer_algorithm', type = str, default = 'herding',
                        help='The algorithm for updating buffer (default: herding).')
    return parser

class CRR(ModifiedContinualModel):
    """
    Continual relation-reasoning Network, to continually learn and reason the relation of two images.
    """
    NAME = 'crr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self,  backbone, loss, args, transform , projection : bool = False, feat_dim_projection: int = 128, feat_dim_relation: int = 128, aggregation: str = 'cat') -> None:
        """
        Instantiates the network.
        Args:
            backbone: the backbone of the model (ResNet)
            loss: the loss class using for the model.
            args: list of arguments of the model, such as learning rate.
            transform: augmentation of the data.
            
            projection: whether feature after going to backbone would be projected into lower representation space or not.
            feat_dim_projection: the size of projection space.
            feat_dim_relation: the number of nodes in the middle layer of Relation Module.
            aggregation: 4 types of aggregation of two vectors (cat, mean, max, sum).
        Return:
            None
        """
        super(CRR, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.dataset = get_dataset(args)
        self.projection = projection

        self.feat_dim_projection = feat_dim_projection
        if (projection == False):
            self.feat_dim_projection = self.dim_in

        self.feat_dim_relation = feat_dim_relation
        self.aggregation = aggregation

        self.head = nn.Sequential(
            nn.Dropout(p = 0.05),
            nn.Linear(self.dim_in, self.feat_dim_projection),
            nn.ReLU(inplace=True),
        )
        if (self.aggregation == 'cat'): resizer = 2
        elif (self.aggregation == 'sum'): resizer = 1
        elif (self.aggregation == 'mean'): resizer = 1
        elif (self.aggregation == 'max'): resizer = 1
        else: 
            raise NotImplementedError(
                'This type of aggregation function is not supported: {}'.format(self.aggregation))

        # self.conv1 = nn.Conv1d(1, 1, 16, 2)
        # self.bn1 = nn.BatchNorm1d(505)

        self.relation_module = nn.Sequential(
            nn.Dropout(p = 0.05),

            nn.Linear(self.feat_dim_projection * resizer, self.feat_dim_relation),
            nn.BatchNorm1d(self.feat_dim_relation),
            nn.LeakyReLU(),
            
            nn.Linear(self.feat_dim_relation, 16),
        )
        self.decision = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )
        self.class_means = None
        self.best_prototype = {}

        self.old_net = None
        self.current_task = 0

        self.scheduler = MultiStepLR(self.opt, milestones=[20, 40, 70, 90, 120, 140, 170, 190, 220, 240], gamma=0.8) #verbose = True)
    
    def get_scheduler(self):
        return self.scheduler
    
    def get_opt(self):
        return self.opt

    def aggregate(self, feature_vector1: torch.Tensor, feature_vector2: torch.Tensor, type: str = 'cat') -> torch.Tensor:
        """
        Aggregate function of two representation vectors.
        Args:
            feature_vector1: representation features of the first image.
            feature_vector2: representation features of the second image.
            type: cat, sum, mean, max.
        Return:
            aggregation_vector: the aggregation features of 2 images.
        """
        if (type == 'cat'):
            agg_feat = torch.cat((feature_vector1, feature_vector2), 1)
        elif (type == 'sum'):
            agg_feat = feature_vector1 + feature_vector2
        elif (type == 'mean'):
            agg_feat = (feature_vector1 + feature_vector2) / 2.0
        elif (type == 'max'):
            agg_feat, _ = torch.max(torch.stack((feature_vector1, feature_vector2), 2), dim = 2)

        return agg_feat

    def forward_one(self, x : torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass of an input.
        Args:
            x: the input tensor.
        Return:
            the output tensor (feature vector) after going through the backbones. (feature extractors).
        """
        x = self.net(x)
        #x = x.view(x.size()[0], -1)
        if (self.projection == True): 
            x = F.normalize(self.head(x), dim=1)
        return x

    def reasoning_features(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute reasoning features of two given input x1, x2.
        Args:
            x1: the first input tensor. N-x1, N-x2.
            x2: the second input tensor.
        Return:
            reason features (before going through decision layer) 
        """
        feat_vec1 = self.forward_one(x1)
        feat_vec2 = self.forward_one(x2)

        feat_vec1 = F.normalize(feat_vec1, dim=1)
        feat_vec2 = F.normalize(feat_vec2, dim=1)

        agg_feat = self.aggregate(feat_vec1, feat_vec2, self.aggregation)
        # out = agg_feat.unsqueeze(1)
        
        # out = self.conv1(out)
        # out = nn.LeakyReLU()(out)
        # out = out.squeeze()
        # out = self.bn1(out)

        out = self.relation_module(agg_feat)

        return out

    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, test: bool = False) -> torch.Tensor:
        """
        Compute a forward pass of two inputs.
        Args:
            x1: the first input tensor. N-x1, N-x2.
            x2: the second input tensor.
        Return:
            the probability that x1 and x2 belong to the same class (output tensor).
        """
        feat_vec1 = self.forward_one(x1)
        feat_vec2 = self.forward_one(x2)

        feat_vec1 = F.normalize(feat_vec1, dim=1)
        feat_vec2 = F.normalize(feat_vec2, dim=1)

        agg_feat = self.aggregate(feat_vec1, feat_vec2, self.aggregation)
        # out = agg_feat.unsqueeze(1)
        
        # out = self.conv1(out)
        # out = nn.LeakyReLU()(out)
        # out = out.squeeze()
        # out = self.bn1(out)

        out = self.relation_module(agg_feat)

        out = self.decision(out)
        
        if test == True:
            out = torch.sigmoid(out)

        return out

    def forward_test(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        compute the outputs for the images
        Args:
            inputs: input batch tensor.
        Return:
            y: which classes these images belong to.
        """
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels = self.buffer.get_all_data(transform)
        classes_sorted, _ = torch.sort(self.classes_so_far)
        output_all_classes = None 

        for _y in classes_sorted:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)

            idx = torch.randint(0, x_buf.shape[0], (min(x_buf.shape[0], 8), ))
            x_buf = x_buf[idx]
            
            new_inputs = torch.repeat_interleave(inputs, x_buf.shape[0], dim = 0)
            new_x_buf = torch.cat(inputs.shape[0] * [x_buf], dim = 0)
            
            output_each_class = self.forward(new_inputs, new_x_buf)
            output_each_class = output_each_class.view((inputs.shape[0], -1))

            output_each_class = torch.mean((output_each_class), dim = 1, keepdim = True)

            if (output_all_classes is not None):
                #print(f'before:{output_all_classes.shape}')
                #print(f'here: {output_each_class.shape}')
                output_all_classes = torch.cat([output_all_classes, output_each_class], dim = 1)

            else:
                output_all_classes = output_each_class
            
            #print(f'after:{output_all_classes.shape}')
            
            del x_buf
            gc.collect()
            torch.cuda.empty_cache()
        
        output_all_classes = output_all_classes.squeeze()
        #y = torch.argmax(output_all_classes, dim=1)

        #self.class_means = torch.stack(class_means)
        return output_all_classes

    def crr_collate(self, inputs: torch.Tensor, labels: torch.Tensor = None, old_inputs: torch.Tensor = None, old_labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = inputs.shape[0]

        #Just pairs of current class with current one.
        idx_1 = torch.randperm(inputs.shape[0])
        idx_2 = torch.randperm(inputs.shape[0])

        inputs_1 = torch.clone(inputs)[idx_1].to(self.device)
        inputs_2 = torch.clone(inputs)[idx_2].to(self.device)
        labels_pair = (labels[idx_1] == labels[idx_2]).long().to(self.device)

        # pairs of current class with old one.
        if (old_inputs is not None):
            idx_1 = torch.randperm(inputs.shape[0])
            inputs_1 = torch.cat([inputs_1, inputs[idx_1]], dim = 0)

            #rand = torch.randint(0, old_inputs.shape[0], (inputs.shape[0], ))
            rand = torch.randperm(old_inputs.shape[0])[:inputs.shape[0]]

            inputs_2 = torch.cat([inputs_2, old_inputs[rand]], dim = 0)

            labels_pair = torch.cat([labels_pair, (labels[idx_1] == old_labels[rand]).long()], dim = 0)
        else:
            idx_1 = torch.randperm(inputs.shape[0])
            idx_2 = torch.randperm(inputs.shape[0])

            inputs_1 = torch.cat([inputs_1, inputs[idx_1]], dim = 0)
            inputs_2 = torch.cat([inputs_2, inputs[idx_2]], dim = 0)

            labels_pair = torch.cat([labels_pair, (labels[idx_1] == labels[idx_2]).long()], dim = 0)

        return inputs_1, inputs_2, labels_pair

    def get_buffer(self):
        return self.buffer
    
    def get_prototypes(self, k : int = 20):
        assert self.args.buffer_algorithm == 'mmd'
        
        prototypes, prototypes_label = None, None

        buf_x, buf_y = self.buffer.get_all_data()

        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y = buf_x[idx], buf_y[idx]

            idx = torch.randperm(_y_y.shape[0])[:k] #torch.randint(0, _y_y.shape[0], (samples_per_class, ))
                
            _y_x = _y_x[idx]
            _y_y = _y_y[idx]

            if (prototypes is None):
                prototypes = _y_x
                prototypes_label = _y_y
            else:
                prototypes = torch.cat([prototypes, _y_x], dim = 0)
                prototypes_label = torch.cat([prototypes_label, _y_y], dim = 0)
            
        return prototypes, prototypes_label

    def get_class_means(self):
        return self.class_means
    
    def get_best_prototype(self):
        return self.best_prototype

    def step_scheduler(self, steps: int) -> None :
        self.scheduler.step(steps)

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        class_means = []

        transform = self.dataset.get_normalization_transform()
        examples, labels = self.buffer.get_all_data(transform)
        classes_sorted, _ = torch.sort(self.classes_so_far)

        for _y in classes_sorted:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)

            feats = self.forward_one(x_buf)
            for j in range(feats.size(0)):  # Normalize
                feats.data[j] = feats.data[j] / feats.data[j].norm()

            feats = feats.to('cpu')
            mean_feats = feats.mean(0)
            mean_feats.data = mean_feats.data/ mean_feats.data.norm()
            
            # 1. Choose the prototypes - mmd algorithm
            # gamma = self.args.gamma
            # kernel_type = 'global' # 'global, local'
            # num_prototypes = 2

            # d_train = VectorEmbeddingDataset(feats, torch.tensor([_y]*feats.shape[0]), None)
            # if kernel_type == 'global':
            #     d_train.compute_rbf_kernel(gamma)
            # elif kernel_type == 'local':
            #     d_train.compute_local_rbf_kernel(gamma)
            # else:
            #     raise KeyError('kernel_type must be either "global" or "local"')
                
            # prototype_indices = select_prototypes(d_train.K, num_prototypes)
            # prototypes = d_train.X[prototype_indices]

            # 2. Choose the prototypes - k-means algorithm.

            #kmeans = KMeans(n_clusters=2, random_state=0).fit(feats)
            #prototypes = torch.tensor(kmeans.cluster_centers_)

            # Dark magic

            #mean_feats = 0.7 * mean_feats + 0.15 * prototypes[0] + 0.15 * prototypes[1]
        
            class_means.append(mean_feats)

            del x_buf, feats
            gc.collect()
            torch.cuda.empty_cache()

        #self.class_means = torch.stack(class_means)

        self.class_means = class_means

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor , not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        Args:
            inputs: batch of examples
            labels: ground-truth labels
        Return: 
            the value of the loss function
        """
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_inputs, buf_labels = buf_inputs.to(self.device), buf_labels.to(self.device) 
            
            #inputs = torch.cat((inputs, buf_inputs))
            #labels = torch.cat((labels, buf_labels))
        else:
            buf_inputs, buf_labels = None, None

        inputs_1, inputs_2, labels_pair = self.crr_collate(inputs, labels, buf_inputs, buf_labels)
        #inputs_1, inputs_2, labels_pair = inputs_1.to(self.device), inputs_2.to(self.device), labels_pair.to(self.device)
        
        outputs = self.forward(inputs_1, inputs_2)
        outputs = torch.squeeze(outputs)
        labels_pair = labels_pair.type_as(outputs)

        loss = self.loss(outputs, labels_pair) #F.mse_loss(torch.sigmoid(outputs), labels_pair) #self.loss(outputs, labels_pair)

        # Replay on old pairs
        if (buf_inputs is not None):
            old_inputs_1, old_inputs_2, old_labels_pair = self.crr_collate(buf_inputs, buf_labels)
            #old_inputs_1, old_inputs_2, old_labels_pair = old_inputs_1.to(self.device), old_inputs_2.to(self.device), old_labels_pair.to(self.device)
            
            old_outputs = self.forward(old_inputs_1, old_inputs_2)
            old_outputs = torch.squeeze(old_outputs)

            if (self.old_net is not None):
                # old_model_output = self.old_net.forward(old_inputs_1, old_inputs_2)
                # old_model_output = torch.squeeze(old_model_output)             
                # loss += self.args.beta * self.loss(old_outputs, torch.sigmoid(old_model_output))
                
                # loss += self.args.beta * F.mse_loss(torch.sigmoid(old_outputs), torch.sigmoid(old_model_output))

                old_model_reasoning_features = self.old_net.reasoning_features(old_inputs_1, old_inputs_2)
                old_model_reasoning_features = torch.squeeze(old_model_reasoning_features)             
                current_model_reasoning_features = self.reasoning_features(old_inputs_1, old_inputs_2)
                current_model_reasoning_features = torch.squeeze(current_model_reasoning_features)  
                
                temp = self.args.temp
                loss += self.args.beta * nn.KLDivLoss()(F.log_softmax(old_model_reasoning_features/temp, dim=1), F.softmax(current_model_reasoning_features/temp, dim=1)) * (temp*temp)

            old_labels_pair = old_labels_pair.type_as(old_outputs)
            loss += self.args.alpha * self.loss(old_outputs, old_labels_pair) #F.mse_loss(torch.sigmoid(old_outputs), old_labels_pair) #self.loss(old_outputs, old_labels_pair)

            del old_inputs_1, old_inputs_2, old_labels_pair
            torch.cuda.empty_cache()
            gc.collect()

        loss.backward()
        self.opt.step()

        #self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size]) #herding (already converted to it). 
        del inputs_1, inputs_2, labels_pair, buf_inputs, buf_labels
        gc.collect()
        torch.cuda.empty_cache()
        
        return loss.item()

    def end_task(self, dataset) -> None:
        self.eval()
        self.old_net = deepcopy(self)
        self.train()
        with torch.no_grad():
            #self.update_k_best_prototypes(dataset, 5)
            if (self.args.buffer_algorithm == 'herding'):
                self.fill_buffer(self.buffer, dataset, self.current_task)
            elif (self.args.buffer_algorithm == 'mmd'):
                self.fill_buffer_mmd(self.buffer, dataset, self.current_task)
        self.current_task += 1
        self.eval()
        with torch.no_grad():
            self.compute_class_means()
        self.train()

    def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy. #Adapted from ICaRL.py
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        self.eval()
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y = self.buffer.get_all_data()
            mem_buffer.empty()

            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y = buf_x[idx], buf_y[idx]
                mem_buffer.add_data(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class],
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1 Extract all features
        a_x, a_y, a_f = [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))

            feats = self.forward_one(x)
            for j in range(feats.size(0)):  # Normalize
                feats.data[j] = feats.data[j] / feats.data[j].norm()

            a_f.append(feats.cpu())

            del x, not_norm_x, y, feats
            gc.collect()
            torch.cuda.empty_cache()

        a_x, a_y, a_f = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f)

        # 2.2 Compute class means
        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y = a_x[idx], a_y[idx]
            feats = a_f[idx]

            # Algorithm 1: Randomly choosing equally samples from each class.
            # storing_nums = min(feats.shape[0], samples_per_class)
            # if samples_per_class >= feats.shape[0]:
            #     mem_buffer.add_data(
            #         examples=_x[0:feats.shape[0]].to(self.device),
            #         labels=_y[0:feats.shape[0]].to(self.device),
            #     )
            # else:
            #     idx_choose = torch.randint(0, feats.shape[0], (samples_per_class, ))
            #     mem_buffer.add_data(
            #         examples=_x[idx_choose].to(self.device),
            #         labels=_y[idx_choose].to(self.device),
            #     )

            # Algorithm 2: Herding Algorithm
            mean_feat = feats.mean(0)
            mean_feat = mean_feat.data/ mean_feat.data.norm() # Normalize
            mean_feat = torch.unsqueeze(mean_feat, 0)

            running_sum = torch.zeros_like(mean_feat)
            i = 0
            while i < samples_per_class and i < feats.shape[0]:
                cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

                idx_min = cost.argmin().item()

                mem_buffer.add_data(
                    examples=_x[idx_min:idx_min + 1],
                    labels=_y[idx_min:idx_min + 1],
                )

                running_sum += feats[idx_min:idx_min + 1]
                feats[idx_min] = feats[idx_min] + 1e6
                i += 1

        assert len(mem_buffer.examples) <= mem_buffer.buffer_size

        print("Updated the buffer successfully!")

        self.train()

    def update_k_best_prototypes(self, dataset, k: int = 2) -> torch.Tensor:
        self.eval()

        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1. Extract all features
        a_x, a_y, a_f = [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))

            feats = self.forward_one(x)
            for j in range(feats.size(0)):  # Normalize
                feats.data[j] = feats.data[j] / feats.data[j].norm()

            a_f.append(feats.cpu())

            del x, not_norm_x, y, feats
            gc.collect()
            torch.cuda.empty_cache()

        a_x, a_y, a_f = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f)

        gamma = self.args.gamma

        kernel_type = 'global' # 'global, local'

        # regularizer = None
        regularizer = 'logdet'
        # regularizer = 'iterative'
        num_prototypes = k

        # 2.2. maximum mean discrepancy algorithm
        for _y_class in a_y.unique():
            idx = (a_y == _y_class)
            _x, _y = a_x[idx], a_y[idx]
            feats = a_f[idx]

            d_train = VectorEmbeddingDataset(feats, _y, _x)
            if kernel_type == 'global':
                d_train.compute_rbf_kernel(gamma)
            elif kernel_type == 'local':
                d_train.compute_local_rbf_kernel(gamma)
            else:
                raise KeyError('kernel_type must be either "global" or "local"')
            
            # 2.2.1 Select prototypes
            prototype_indices = select_prototypes(d_train.K, num_prototypes)

            prototypes = d_train.X[prototype_indices]
            prototype_labels = d_train.y[prototype_indices]

            self.best_prototype[int(_y_class)] = prototypes

        self.train()

    def fill_buffer_mmd(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        """
        Adds examples from the current task to the memory buffer
        by mmd method (choosing prototypes and criticisms)
        Paper: Examples are not Enough, Learn to Criticize! Criticism for Interpretability [NIPS 2016]
        """
        
        self.eval()
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

        # 1) First, subsample prior classes (RANDOMLY)
        if t_idx > 0:
            buf_x, buf_y = self.buffer.get_all_data()
            mem_buffer.empty()

            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y = buf_x[idx], buf_y[idx]

                idx = torch.randperm(_y_y.shape[0])[:samples_per_class] #torch.randint(0, _y_y.shape[0], (samples_per_class, ))
                
                _y_x = _y_x[idx]
                _y_y = _y_y[idx]

                mem_buffer.add_data(
                    examples=_y_x,
                    labels=_y_y,
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1. Extract all features
        a_x, a_y, a_f = [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))

            feats = self.forward_one(x)
            for j in range(feats.size(0)):  # Normalize
                feats.data[j] = feats.data[j] / feats.data[j].norm()

            a_f.append(feats.cpu())

            del x, not_norm_x, y, feats
            gc.collect()
            torch.cuda.empty_cache()

        a_x, a_y, a_f = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f)

        gamma = self.args.gamma

        kernel_type = 'global' # 'global, local'

        # regularizer = None
        regularizer = 'logdet'
        # regularizer = 'iterative'

        # 2.2. maximum mean discrepancy algorithm
        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y = a_x[idx], a_y[idx]
            feats = a_f[idx]
            
            num_prototypes = min(2 * samples_per_class // 3, 2 * feats.shape[0] // 3)
            num_criticisms = min(samples_per_class - 2 * samples_per_class // 3, feats.shape[0] - 2 * feats.shape[0] // 3)

            d_train = VectorEmbeddingDataset(feats, _y, _x)
            if kernel_type == 'global':
                d_train.compute_rbf_kernel(gamma)
            elif kernel_type == 'local':
                d_train.compute_local_rbf_kernel(gamma)
            else:
                raise KeyError('kernel_type must be either "global" or "local"')
            
            # 2.2.1 Select prototypes
            prototype_indices = select_prototypes(d_train.K, num_prototypes)

            prototypes = d_train.Z[prototype_indices]
            prototype_labels = d_train.y[prototype_indices]

            mem_buffer.add_data(
                    examples=prototypes,
                    labels=prototype_labels,
                )

            # 2.2.2 Select criticisms
            criticism_indices = select_criticisms(d_train.K, prototype_indices, num_criticisms, regularizer)

            criticisms = d_train.Z[criticism_indices]
            criticism_labels = d_train.y[criticism_indices]

            mem_buffer.add_data(
                    examples=criticisms,
                    labels=criticism_labels,
                )
            
            assert prototype_labels.shape[0] + criticism_labels.shape[0] == min(samples_per_class, feats.shape[0])

        assert len(mem_buffer.examples) <= mem_buffer.buffer_size
        self.train()
