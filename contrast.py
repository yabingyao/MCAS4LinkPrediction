#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pythonProject2 
@File    ：contrast.py
@IDE     ：PyCharm 
@Author  ：guopingxia 
@Date    ：2024-02-28 9:06 
'''
from re import L

import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse

from GCL.models import SingleBranchContrast
from torch import device
from torch_geometric.graphgym import cmd_args

sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main1 import *
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net, GConv, FC, Encoder
import numpy as np
import torch
import torch.nn.functional as F


class JointLoss(torch.nn.Module):
    """
    When computing loss, we are using a 2Nx2N similarity matrix, in which positve samples are on the diagonal of four
    quadrants while negatives are all the other samples.

    Parameters
    ----------
    config : Dict
        Configuration options for the JointLoss.

    Attributes
    ----------
    batch_size : int/50
        Number of nodes in the graph.
    temperature : float
        Temperature to use scale logits.
    device : str
        Device to use: GPU or CPU.
    softmax : torch.nn.Softmax
        Softmax layer for generating probabilities.
    mask_for_neg_samples : torch.Tensor
        Mask to use to get negative samples from similarity matrix.
    similarity_fn : Callable
        Function to generate similarity matrix: Cosine, or Dot product.
    criterion : torch.nn.CrossEntropyLoss
        Loss function for training.
    """
    def __init__(self, config):
        super(JointLoss, self).__init__()
        # Assign config to self
        self.config = config
        # Batch size == number of nodes in the graph
        self.batch_size = config["out_dim"] # Not used --- overwritten in forward() method
        # Temperature to use scale logits
        self.temperature = config["tau"]
        # Device to use: GPU or CPU
        self.device = config["device"]
        # initialize softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._cosine_simililarity if config["cosine_similarity"] else self._dot_simililarity
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_mask_for_neg_samples(self) -> torch.Tensor:
        """
        Generates a mask for selecting negative samples.

        Returns
        -------
        torch.Tensor
            Mask for selecting negative samples.
        """
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = torch.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(torch.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the dot product similarity between two tensors.

        Parameters
        ----------
        x : torch.Tensor
        y : torch.Tensor

        Returns
        -------
        torch.Tensor
            Dot product similarity matrix.
        """
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = torch.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between two tensors.

        Parameters
        ----------
        x : torch.Tensor
        y : torch.Tensor

        Returns
        -------
        torch.Tensor
            Cosine similarity matrix.
        """
        similarity = torch.nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)

    def XNegloss(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Computes the XNeg loss.

        Parameters
        ----------
        representation : torch.Tensor
            Tensor representing the representations.

        Returns
        -------
        torch.Tensor
            XNeg loss.
        """
        # 计算相似性矩阵
        similarity = self.similarity_fn(representation, representation)
        # 正样本对的相似性分数
        l_pos = torch.diag(similarity, self.batch_size)
        # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
        r_pos = torch.diag(similarity, -self.batch_size)
        # 拼接所有正样本对 as a 2nx1 column vector
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # 负样本对的相似性分数 (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = torch.cat((positives, negatives), dim=1)
        # 通过温度系数归一化logits
        logits /= self.temperature
        # Labels are all zeros since all positive samples are the 0th column in logits array.
        # So we will select positive samples as numerator in NTXentLoss
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # Compute total loss
        loss = self.criterion(logits, labels)
        # Loss per sample
        closs = loss / (2 * self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, representation: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward propagation method for the JointLoss class.

        Parameters
        ----------
        representation : torch.FloatTensor
            The representation matrix of nodes in the graph.

        Returns
        -------
        loss : torch.FloatTensor
            The contrastive loss for the representation.

        """
        if self.config["dataset"] == "YST":
            representation = representation.cpu()
            self.device = representation.device
        
        # Overwrite batch_size as half of the first dimension of representation
        self.batch_size = int(representation.size(0)//2)

        # Initialize loss
        loss = None
        
        if self.config["contrastive_loss"]:
            
            # Mask to use to get negative samples from similarity matrix
            self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        
            closs = self.XNegloss(representation)
            loss = closs

        # Return
        return loss
