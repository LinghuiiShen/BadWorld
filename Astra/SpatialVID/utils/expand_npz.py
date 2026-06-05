"""
Mask utility functions for processing sparse matrix data.
"""

import numpy as np
from scipy.sparse import csr_matrix


def expand(loaded_data):
    """
    Reconstruct 3D mask from sparse matrix data.
    
    Args:
        loaded_data (dict): Dictionary containing sparse matrix data with keys:
            - 'shape': Original matrix dimensions
            - 'f_{i}_data': Sparse matrix data for frame i
            - 'f_{i}_indices': Sparse matrix indices for frame i  
            - 'f_{i}_indptr': Sparse matrix index pointers for frame i
    
    Returns:
        np.ndarray: 3D array with shape (frames, height, width)
    """
    reconstructed_sparse_matrices = []
    num_frames = (len(loaded_data) - 1) // 3  # Calculate number of frames
    matrix_shape = loaded_data['shape']  # Get original matrix dimensions

    # Reconstruct sparse matrix for each frame
    for i in range(num_frames):
        data = loaded_data[f'f_{i}_data']
        indices = loaded_data[f'f_{i}_indices']
        indptr = loaded_data[f'f_{i}_indptr']
        reconstructed_matrix = csr_matrix((data, indices, indptr), shape=matrix_shape)
        reconstructed_sparse_matrices.append(reconstructed_matrix)

    # Stack all frames into a 3D array (frames, height, width)
    reconstructed_mask_3d = np.stack([m.toarray() for m in reconstructed_sparse_matrices], axis=0)
    return reconstructed_mask_3d
