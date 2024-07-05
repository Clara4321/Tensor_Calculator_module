# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 22:30:04 2024

@author: Clara4321
"""

import torch

__all__ = ['TensorCalculator']

class TensorCalculator():

    def __init__(self):
        pass

    def all_zeros(self, dim_x, dim_y):
        return torch.zeros([dim_x, dim_y])

    def all_ones(self, dim_x, dim_y):
        return torch.ones([dim_x, dim_y])

    def random_tensor(self, dim_x, dim_y):
        return torch.rand([dim_x, dim_y])

    def sum_tensors(self, tensor1, tensor2):
        return torch.add(tensor1, tensor2)

    def multiply_tensors(self, tensor1, tensor2):
        return torch.mul(tensor1, tensor2)

    def transpose_tensor(self, tensor):
        return torch.transpose(tensor, 0, 1)

    def tensor_dot_product(self, tensor1, tensor2):
        return torch.matmul(tensor1, tensor2)
