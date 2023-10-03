import numba.cuda as cuda
import numba as nb
import numpy as np
import math

from numba import float64 as float64
from numba import int64 as int64
from typing import Final
from numba.cuda.cudadrv.devicearray import DeviceNDArray

@cuda.jit(device = True)
def multivariate_gaussian_kernel(
        x_coord: Final[int64], 
        y_coord: Final[int64],
        d_grid: Final[DeviceNDArray], 
        d_covariance_matrix: Final[DeviceNDArray], 
        d_mu: Final[DeviceNDArray], 
        d_sigma: Final[DeviceNDArray]
    ) -> float64:
    """
    (Device Function)
    
    Computes the mutation probability distribution function 
    
    Arguments:
        x_coord: The horizontal index of the d_grid array at which the gaussian kernel is computed
        y_coord: The vertical index of the d_grid array at which the gaussian kernel is computed
        d_grid: The grid containing the x/y-values
        d_covariance_matrix: The covariance matrix of the mutation distribution
        d_mu: The mean vector (current centroid)
        d_sigma: The step size
    Returns:
        The probability density function evaluated at the given point

    """
    sigma = d_sigma.item()
    
    difference_vector = cuda.local.array(shape = 2, dtype = nb.float64)
    for dim in range(2):
        difference_vector[dim] = d_grid[x_coord, y_coord, dim]- d_mu[dim]

    
    det_covariance = (sigma ** 4 * (d_covariance_matrix[0, 0] * d_covariance_matrix[1, 1] - d_covariance_matrix[1, 0] * d_covariance_matrix[0, 1]))
    
    norm_denominator = math.sqrt(det_covariance)  * (2 * math.pi)

    norm = 1 / norm_denominator

    ex_arg = cuda.local.array(shape = 2, dtype = nb.float64)

    ex_arg[0] += (difference_vector[0] * sigma ** 2 * d_covariance_matrix[1, 1] + difference_vector[1] * sigma ** 2 * -d_covariance_matrix[1,0]) / det_covariance

    ex_arg[1] += (difference_vector[0] * sigma ** 2 * -d_covariance_matrix[0, 1] + difference_vector[1] * sigma ** 2 *d_covariance_matrix[0,0]) / det_covariance
    
    ex_arg_scalar = ex_arg[0] * difference_vector[0] + ex_arg[1] * difference_vector[1]

    return norm * math.exp(-0.5 * ex_arg_scalar)

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

def compute_mutation_probability_map(
        grid: np.ndarray, 
        n_points: int,
        covariance_matrix: np.ndarray, 
        sigma: float, 
        mu: np.ndarray
    ):

    """
    (Host function)

    Interface on the host for computing the mutation probability map on the grid

    
    Arguments:
        grid: The grid containing the x/y-values
        n_point: The number of points which to construct the grid with
        covariance_matrix: The covariance matrix of the mutation distribution
        mu: The mean vector (current centroid)
        sigma: The step size


    """
    output_array = np.empty((n_points, n_points), dtype = np.float64)
    d_grid = cuda.to_device(grid)
    d_output_array = cuda.to_device(output_array)
    d_covariance_matrix = cuda.to_device(covariance_matrix)
    d_sigma = cuda.to_device(sigma)
    d_mu = cuda.to_device(mu)

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(n_points / threads_per_block[0])
    blocks_per_grid_y = math.ceil(n_points / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    cuda.synchronize()
    gaussian_kernel_cuda[blocks_per_grid, threads_per_block](
        d_grid, 
        d_output_array,
        d_covariance_matrix, 
        d_mu, 
        d_sigma
    ) 
    cuda.synchronize()
    return d_output_array.copy_to_host()