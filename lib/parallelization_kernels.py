#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

A simple script containing some simple utility functions for 
parallelizing the computation of the mutation distribution density function
for plotting the runs of the CMA algorithm

Contains:

* 1) The device function, `__multivariate_gaussian_kernel`,which is executed on the gpu
* 2) The kernel function, `__gaussian_kernel_cuda` that calls the device function
* 3) The host function, `compute_mutation_probability_map` that calls the kernel function and writes the results to the host (cpu)

"""

from .parallelization_device_functions import *

@cuda.jit()
def gaussian_kernel_cuda(
        d_grid: Final[DeviceNDArray], 
        d_output_array: DeviceNDArray,
        d_covariance_matrix: Final[DeviceNDArray], 
        d_mu: Final[DeviceNDArray], 
        d_sigma: Final[float64]
    ) -> None:
    """
    (Kernel)

    Computes the mutation probability distribution function 

    Arguments:
        d_grid: The grid containing the x/y-values
        d_output_array: The array which to write the results to 
        d_covariance_matrix: The covariance matrix of the mutation distribution
        d_mu: The mean vector (current centroid)
        d_sigma: The step size
    Returns:
        The mapping probability density function evaluated on d_grid
    """
    image_width, image_height, n_dim = d_grid.shape
    x_coord, y_coord = cuda.grid(2)
    if x_coord < image_width and y_coord < image_height:
        d_output_array[x_coord, y_coord] = multivariate_gaussian_kernel(
            x_coord, 
            y_coord,
            d_grid, 
            d_covariance_matrix, 
            d_mu, 
            d_sigma
        )
