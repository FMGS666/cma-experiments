from .parallelization_kernels import *

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

    print(f"{mu=}")
    print(f"{sigma=}")
    print(f"{covariance_matrix=}")

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
