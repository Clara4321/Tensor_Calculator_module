# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:36:28 2024

@author: Clara4321
"""

from tensor_operations_module import TensorCalculator

# Initialize the TensorCalculator
calculator = TensorCalculator()

# Create tensors
tensor1 = calculator.all_zeros(2, 2)
tensor2 = calculator.all_ones(2, 2)
tensor3 = calculator.random_tensor(2, 2)

# Tensor operations
sum_result = calculator.sum_tensors(tensor1, tensor2)
multiply_result = calculator.multiply_tensors(tensor1, tensor2)
transpose_result = calculator.transpose_tensor(tensor3)
dot_product_result = calculator.tensor_dot_product(tensor1, tensor2)

# Print results
print("Sum of tensors:\n", sum_result)
print("Multiplication of tensors:\n", multiply_result)
print("Transpose of tensor:\n", transpose_result)
print("Dot product of tensors:\n", dot_product_result)



