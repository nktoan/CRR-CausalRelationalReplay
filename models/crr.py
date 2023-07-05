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
import random

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
    parser = ArgumentParser(description='Class-Incremental Learning'
                                        'with Causal Relational Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default = 0.75,
                        help='Penalty weight for current pairs (intrinsic relation).')
    parser.add_argument('--beta', type=float, default = 1.75,
                        help='Penalty weight for old pairs replay.')
    parser.add_argument('--gamma', type=float, default = 1.25,
                        help='Penalty weight for old pairs (intrinsic relation).')
    parser.add_argument('--end_lr', type=float, default = 0.42,
                        help='The final learning rate after n tasks.')

    parser.add_argument('--buffer_algorithm', type = str, default = 'herding',
                        help='The algorithm for updating buffer (default: herding).')
    return parser

class CRR(ModifiedContinualModel):
    """
    Causal Relational Replay
    """
    NAME = 'crr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self,  backbone, loss, args, transform , projection : bool = True, feat_dim_projection: int = 128, feat_dim_relation: int = 16, aggregation: str = 'cat') -> None:
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

        self.projection_head = nn.Sequential( #512 -> 128
            nn.Linear(self.dim_in, self.dim_in),
            nn.BatchNorm1d(self.dim_in),
            nn.ReLU(inplace = True),
            nn.Linear(self.dim_in, self.feat_dim_projection),
            nn.BatchNorm1d(self.feat_dim_projection),
        )

        self.criterion_loss = nn.CrossEntropyLoss()

        if (self.aggregation == 'cat'): self.resizer = 2
        elif (self.aggregation == 'sum'): self.resizer = 1
        elif (self.aggregation == 'mean'): self.resizer = 1
        elif (self.aggregation == 'max'): self.resizer = 1
        else:
            raise NotImplementedError(
                'This type of aggregation function is not supported: {}'.format(self.aggregation))

        # self.conv1 = nn.Conv1d(1, 1, 16, 2)
        # self.bn1 = nn.BatchNorm1d(505)

        self.relation_module = nn.Sequential(
            nn.Dropout(p = 0.07),
            nn.Linear(self.feat_dim_projection * self.resizer, self.feat_dim_relation),
            nn.BatchNorm1d(self.feat_dim_relation),
            nn.LeakyReLU(),
        )
        self.decision = nn.Sequential(
            nn.Linear(self.feat_dim_relation, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_in, 10),
        )
        self.class_means = None
        self.best_prototype = {}

        self.old_net = None
        self.current_task = 0

        self.opt = SGD([{'params': self.relation_module.parameters()},
                        {'params': self.decision.parameters()},
                        {'params': self.net.parameters()},
                        {'params': self.projection_head.parameters()}], lr=self.args.lr)
        
        if (self.args.dataset == 'seq-cifar10'): #5 tasks - 50e
            start_lr, end_lr, n_step = self.args.lr, self.args.end_lr, 10
            self.scheduler = MultiStepLR(self.opt, 
                                    milestones=[t for sub_list in [[i + 20, i + 40] for i in range(0, 250, 50)] for t in sub_list], gamma=(end_lr/start_lr)**(1/n_step)) #verbose = True)

        elif (self.args.dataset == 'seq-cifar100'): #10 tasks - 50e
            start_lr, end_lr, n_step = self.args.lr, self.args.end_lr, 20
            self.scheduler = MultiStepLR(self.opt, 
                                    milestones=[t for sub_list in [[i + 20, i + 40] for i in range(0, 500, 50)] for t in sub_list], gamma=(end_lr/start_lr)**(1/n_step)) #verbose = True)

        # elif (self.args.dataset == 'seq-cifar100-20'): #20 tasks - 30e
        #     start_lr, end_lr, n_step = self.args.lr, self.args.end_lr, 40
        #     self.scheduler = MultiStepLR(self.opt, 
        #                             milestones=[t for sub_list in [[i + 15, i + 30] for i in range(0, 800, 40)] for t in sub_list], gamma=(end_lr/start_lr)**(1/n_step)) #verbose = True)

        elif (self.args.dataset == 'seq-core50'): #9 tasks - 20e
            start_lr, end_lr, n_step = self.args.lr, self.args.end_lr, 18
            self.scheduler = MultiStepLR(self.opt, 
                                    milestones=[t for sub_list in [[i + 6, i + 12] for i in range(0, 135, 15)] for t in sub_list], gamma=(end_lr/start_lr)**(1/n_step)) #verbose = True)

        # elif (self.args.dataset == 'seq-tinyimg'): #10 tasks - 70e
        #     start_lr, end_lr, n_step = self.args.lr, self.args.end_lr, 20
        #     self.scheduler = MultiStepLR(self.opt, 
        #                             milestones=[t for sub_list in [[i + 30, i + 60] for i in range(0, 700, 70)] for t in sub_list], gamma=(end_lr/start_lr)**(1/n_step)) #verbose = True)
                                           
        #t for sub_list in [[i + 1] for i in range(0, 10, 2)] for t in sub_list
        
        self.opt2 = SGD(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=1e-4)
        
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
        return F.normalize(self.net(x), dim = 1)
        #x = x.view(x.size()[0], -1)

    def project_one(self, x: torch.Tensor) -> torch.Tensor:
        assert(self.projection == True)
        return F.normalize(self.projection_head(x), dim = 1)

    def make_prediction(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def reasoning_features(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute reasoning features of two given input x1, x2.
        Args:
            x1: the first input tensor. N-x1, N-x2.
            x2: the second input tensor.
        Return:
            reason features (before going through decision layer) 
        """
        #feat_vec1 = self.forward_one(x1)
        #feat_vec2 = self.forward_one(x2)

        agg_feat = self.aggregate(x1, x2, self.aggregation)
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

        agg_feat = self.aggregate(x1, x2, self.aggregation)
        
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
        examples, labels = self.buffer.get_all_data(transform)
        classes_sorted, _ = torch.sort(self.classes_so_far)
        output_all_classes = None

        class_means = self.get_class_means()

        with torch.no_grad():
            new_inputs_emd_vect = self.forward_one(inputs)
            new_inputs_proj_head = self.project_one(new_inputs_emd_vect)

            for _y in classes_sorted:
                class_mean_feats = class_means[_y].unsqueeze(0).to(self.device)
                new_x_buf_proj_head = self.project_one(class_mean_feats)
                
                # mean_feats = new_x_buf_proj_head.mean(0).unsqueeze(0)
                # new_x_buf_proj_head = new_x_buf_proj_head[:n_sample]
                # new_x_buf_proj_head = torch.cat([mean_feats.unsqueeze(0), new_x_buf_proj_head], dim = 0)

                new_inputs = torch.repeat_interleave(new_inputs_proj_head, new_x_buf_proj_head.shape[0], dim = 0)
                new_x_buf = torch.cat(new_inputs_proj_head.shape[0] * [new_x_buf_proj_head], dim = 0)

                output_each_class = self.forward(new_inputs, new_x_buf, test = True)
                output_each_class = output_each_class.view((inputs.shape[0], -1))

                output_each_class = torch.mean((output_each_class), dim = 1, keepdim = True)

                output_each_class = output_each_class.to('cpu')

                if (output_all_classes is not None):
                    #print(f'before:{output_all_classes.shape}')
                    #print(f'here: {output_each_class.shape}')
                    output_all_classes = torch.cat([output_all_classes, output_each_class], dim = 1)

                else:
                    output_all_classes = output_each_class
                
                #print(f'after:{output_all_classes.shape}')
                
                del new_inputs, new_x_buf, class_mean_feats, new_x_buf_proj_head, output_each_class
                gc.collect()
                torch.cuda.empty_cache()
            
            del new_inputs_emd_vect, new_inputs_proj_head
            gc.collect()
            torch.cuda.empty_cache()
        
        output_all_classes = output_all_classes.squeeze()
        
        sum_diff = torch.sum(output_all_classes, axis = 1)
        output_diff_all_classes = torch.clone(output_all_classes)

        for i in range(output_all_classes.shape[1]):
            output_diff_all_classes[:, i] =  1 - (sum_diff - output_all_classes[:, i]) / (output_all_classes.shape[1] - 1)

        output_all_classes = 0.5 * output_all_classes + 0.5 * output_diff_all_classes

        #y = torch.argmax(output_all_classes, dim=1)

        del examples, labels
        gc.collect()
        torch.cuda.empty_cache()        

        #self.class_means = torch.stack(class_means)
        return output_all_classes

    def crr_collate(self, inputs: torch.Tensor, labels: torch.Tensor = None, old_inputs: torch.Tensor = None, old_labels: torch.Tensor = None, seed_number: int = 271199) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = inputs.shape[0]

        torch.manual_seed(seed_number)

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
            
            if (inputs.shape[0] == old_inputs.shape[0]):
                rand = torch.randperm(old_inputs.shape[0])[:inputs.shape[0]]
            else:
                rand = torch.randint(0, old_inputs.shape[0], (inputs.shape[0], ))

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

    def observe(self, inputs_aug_i: torch.Tensor, inputs_aug_j: torch.Tensor, labels: torch.Tensor , not_aug_inputs: torch.Tensor) -> float:
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
            if (self.classes_so_far is None):
                self.classes_so_far = labels.to('cpu').unique()
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.train()
        modules = [self.projection_head, self.net, self.relation_module, self.decision, self.classifier]
        for mod in modules:
            for param in mod.parameters():
                param.requires_grad = True

        real_batch_size = inputs_aug_i.shape[0]

        # Reset gradient
        self.opt.zero_grad()

        # Prepare the old inputs from buffer
        if not self.buffer.is_empty():
            buf_aug_inputs_i, buf_aug_inputs_j, buf_labels = self.buffer.get_pair_data(self.args.minibatch_size, transform=self.transform)
            buf_aug_inputs_i, buf_aug_inputs_j, buf_labels = buf_aug_inputs_i.to(self.device), buf_aug_inputs_j.to(self.device), buf_labels.to(self.device) 
        else:
            buf_aug_inputs_i, buf_aug_inputs_j, buf_labels = None, None, None

        # Compute features and projection vectors

        inputs_aug_i, inputs_aug_j = self.forward_one(inputs_aug_i), self.forward_one(inputs_aug_j)
        inputs_aug_i_proj, inputs_aug_j_proj = self.project_one(inputs_aug_i), self.project_one(inputs_aug_j)

        if (buf_labels is not None):
            # if (self.old_net is not None):
            #     # Regularization on invariant representation (optional)
            #     old_model_buf_aug_inputs_i_proj = self.old_net.project_one(self.old_net.forward_one(buf_aug_inputs_i))
            #     old_model_buf_aug_inputs_j_proj = self.old_net.project_one(self.old_net.forward_one(buf_aug_inputs_j))

            buf_aug_inputs_i, buf_aug_inputs_j = self.forward_one(buf_aug_inputs_i), self.forward_one(buf_aug_inputs_j)
            buf_aug_inputs_i_proj, buf_aug_inputs_j_proj = self.project_one(buf_aug_inputs_i), self.project_one(buf_aug_inputs_j)
        else:
            buf_aug_inputs_i, buf_aug_inputs_j = None, None 
            buf_aug_inputs_i_proj, buf_aug_inputs_j_proj = None, None

        # Step 1: Relational Reasoning
        # 1.1. Block gradient of classifier

        modules = [self.classifier]
        for mod in modules:
            for param in mod.parameters():
                param.requires_grad = False

        # if self.current_task > 1:
        #     for mod in [self.relation_module, self.decision]:
        #         for param in mod.parameters():
        #             param.requires_grad = False

        # 1.2. Regularization on invariant representation (optional)

        # regularizer_loss = torch.nn.MSELoss()
        # loss = self.args.alpha * regularizer_loss(inputs_aug_i_proj, inputs_aug_j_proj)

        # if (buf_labels is not None):
        #     loss += self.args.alpha * regularizer_loss(buf_aug_inputs_i_proj, buf_aug_inputs_j_proj)

        #  1.3. Relational Reasoning

        ## Current pair with old pair
        seed_number = random.randint(0, 10000)

        inputs_1, inputs_2, labels_pair = self.crr_collate(inputs_aug_i_proj, labels, buf_aug_inputs_j_proj, buf_labels, seed_number = seed_number)
        outputs_ij = self.forward(inputs_1, inputs_2)
        outputs_ij = torch.squeeze(outputs_ij)
        labels_pair = labels_pair.type_as(outputs_ij)

        loss = self.loss(outputs_ij, labels_pair)

        inputs_1, inputs_2, labels_pair = self.crr_collate(inputs_aug_j_proj, labels, buf_aug_inputs_i_proj, buf_labels, seed_number = seed_number)

        outputs_ji = self.forward(inputs_1, inputs_2)
        outputs_ji = torch.squeeze(outputs_ji)

        # loss += self.loss(outputs_ji, labels_pair)
        loss += self.args.alpha * torch.nn.MSELoss()(torch.sigmoid(outputs_ij), torch.sigmoid(outputs_ji))
        
        ## Replay on old pairs
        
        if (buf_labels is not None):
            seed_number = random.randint(0, 10000)

            old_inputs_1, old_inputs_2, old_labels_pair = self.crr_collate(buf_aug_inputs_i_proj, buf_labels, buf_aug_inputs_j_proj, buf_labels, seed_number = seed_number)
            
            old_outputs_ij = self.forward(old_inputs_1, old_inputs_2)
            old_outputs_ij = torch.squeeze(old_outputs_ij)

            old_inputs_1, old_inputs_2, old_labels_pair = self.crr_collate(buf_aug_inputs_j_proj, buf_labels, buf_aug_inputs_i_proj, buf_labels, seed_number = seed_number)
            
            old_outputs_ji= self.forward(old_inputs_1, old_inputs_2)
            old_outputs_ji = torch.squeeze(old_outputs_ji)

            old_labels_pair = old_labels_pair.type_as(old_outputs_ji)
            loss += self.args.beta * self.loss(old_outputs_ji, old_labels_pair) #F.mse_loss(torch.sigmoid(old_outputs), old_labels_pair) #self.loss(old_outputs, old_labels_pair)

            loss += self.args.gamma * torch.nn.MSELoss()(torch.sigmoid(outputs_ij), torch.sigmoid(outputs_ji))

            """
            -------We do not use knowledge distillation

            if (self.old_net is not None):
                # Regularization on invariant representation (optional)
                # loss += self.args.alpha * self.args.beta * F.mse_loss(old_model_buf_aug_inputs_i_proj, buf_aug_inputs_i_proj)
                # loss += self.args.alpha * self.args.beta * F.mse_loss(old_model_buf_aug_inputs_j_proj, buf_aug_inputs_j_proj)

                old_inputs_1, old_inputs_2, _ = self.crr_collate(
                    old_model_buf_aug_inputs_i_proj, buf_labels, old_model_buf_aug_inputs_j_proj, buf_labels, seed_number = seed_number)
            
                old_model_output_ij = self.old_net.forward(old_inputs_1, old_inputs_2)
                old_model_output_ij = torch.squeeze(old_model_output_ij)

                old_inputs_1, old_inputs_2, _ = self.crr_collate(
                    old_model_buf_aug_inputs_j_proj, buf_labels, old_model_buf_aug_inputs_i_proj, buf_labels, seed_number = seed_number)
            
                old_model_output_ji = self.old_net.forward(old_inputs_1, old_inputs_2)
                old_model_output_ji = torch.squeeze(old_model_output_ji)
                
                loss += self.args.gamma * (F.mse_loss(torch.sigmoid(old_model_output_ji), torch.sigmoid(old_outputs_ij))
                        + F.mse_loss(torch.sigmoid(old_model_output_ij), torch.sigmoid(old_outputs_ji)))
            """

                # old_model_reasoning_features = self.old_net.reasoning_features(old_inputs_1, old_inputs_2)
                # old_model_reasoning_features = torch.squeeze(old_model_reasoning_features)             
                # current_model_reasoning_features = self.reasoning_features(old_inputs_1, old_inputs_2)
                # current_model_reasoning_features = torch.squeeze(current_model_reasoning_features)  
                
                # temp = self.args.temp
                # loss += self.args.beta * nn.KLDivLoss()(F.log_softmax(current_model_reasoning_features/temp, dim=1), F.softmax(old_model_reasoning_features/temp, dim=1)) * (temp*temp)
            
            del old_inputs_1, old_inputs_2, old_labels_pair
            #del old_model_buf_aug_inputs_i_proj, old_model_buf_aug_inputs_j_proj
            torch.cuda.empty_cache()
            gc.collect()

        loss.backward(retain_graph=True)
        self.opt.step()
        
        loss_value = loss.item()
        # Step 2. Train classifier

        # modules_train = [self.classifier]
        # modules_freeze = [self.net, self.relation_module, self.projection_head, self.decision]
        # for mod in modules_freeze + modules_train:
        #     if (mod in modules_train):
        #         for param in mod.parameters():
        #             param.requires_grad = True
        #     else:
        #         for param in mod.parameters():
        #             param.requires_grad = False

        # out_classification_i, out_classification_j = self.classifier(inputs_aug_i), self.classifier(inputs_aug_j)
        # loss2 = self.criterion_loss(out_classification_i, labels) + self.criterion_loss(out_classification_j, labels)

        # if (buf_labels is not None):
        #     out_classification_buf_i, out_classification_buf_j = self.classifier(buf_aug_inputs_i), self.classifier(buf_aug_inputs_j)
        #     loss2 += self.args.beta * (self.criterion_loss(out_classification_buf_i, buf_labels)
        #                         + self.criterion_loss(out_classification_buf_j, buf_labels))   
        
        # self.opt2.zero_grad()
        # loss2.backward()
        # self.opt2.step()
        
        del buf_aug_inputs_i, buf_aug_inputs_j, buf_labels, inputs_1, inputs_2, labels_pair, outputs_ij
        del inputs_aug_i_proj, inputs_aug_j_proj, buf_aug_inputs_i_proj, buf_aug_inputs_j_proj
        del outputs_ji
        gc.collect()
        torch.cuda.empty_cache()
        
        return loss_value

    def end_task(self, dataset) -> None:
        self.eval()
        del self.old_net
        gc.collect()
        torch.cuda.empty_cache()

        self.old_net = None
        self.old_net = deepcopy(self)
        self.train()

        # for g in self.scheduler.optimizer.param_groups:
        #     g['lr'] =  self.args.lr
        # for g in self.opt.param_groups:
        #     g['lr'] =  self.args.lr     

        with torch.no_grad():
            #self.update_k_best_prototypes(dataset, 5)
            if (self.args.buffer_algorithm == 'herding'):
                self.fill_buffer(self.buffer, dataset, self.current_task)
            elif (self.args.buffer_algorithm == 'mmd'):
                self.fill_buffer_mmd(self.buffer, dataset, self.current_task)
            
            self.save_buffer(self.current_task)

        self.current_task += 1
        self.eval()
        with torch.no_grad():
            self.compute_class_means()
        
        self.train()

    def save_buffer(self, t_idx: int, path: str = f'None') -> None:
        buf_x, buf_y = self.buffer.get_all_data()
        
        # torch.save(buf_x, path + f'buf_x_{t_idx}.pt')
        # torch.save(buf_y, path + f'buf_y_{t_idx}.pt')

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
        for x, _, y, not_norm_x in loader:
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
        for x, _, y, not_norm_x in loader:
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
        for x, _, y, not_norm_x in loader:
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
