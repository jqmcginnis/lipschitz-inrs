import os
import torch
import random
import numpy as np
import deepali
import deepali.core.flow as dp
import torch.nn.functional as F

def coordinate_to_voxel_indices(coordinates, dims):
    dims_tensor = torch.tensor(dims, device=coordinates.device, dtype=coordinates.dtype)
    return (coordinates + 1) * (dims_tensor - 1) / 2

def erode_mask_3d(mask, erosion_size=5):
    # Create 3D erosion kernel
    kernel_size = 2 * erosion_size + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=mask.device)
    
    # Add batch and channel dimensions
    mask_float = mask.float().unsqueeze(0).unsqueeze(0)
    
    # Apply convolution
    conv_result = F.conv3d(mask_float, kernel, padding=erosion_size)
    
    # Only keep pixels that were fully covered by the kernel
    eroded = (conv_result == kernel.numel()).squeeze()
    
    return eroded.bool()

def parse_list(param_str, expected_length):
    """
    Parse parameter strings from command line that can be single values or comma-separated lists.
    
    Args:
        param_str: String from command line (e.g., "1.0" or "1.0,0.5,0.2,0.1")
        expected_length: Expected number of values in the list
    
    Returns:
        List of floats with length equal to expected_length
    """
    if ',' in param_str:
        # Parse as comma-separated list
        param_values = [float(x.strip()) for x in param_str.split(',')]
        if len(param_values) != expected_length:
            raise ValueError(f"Number of values ({len(param_values)}) must match "
                           f"expected length ({expected_length})")
        return param_values
    else:
        # Create list of identical values
        param_value = float(param_str)
        return [param_value] * expected_length

def set_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """

    # Note: Using .to(device) or ensuring disp is on the correct device
    # before operations if your script runs on a GPU.
    # The original code snippet doesn't show where `disp` is loaded or
    # moved to a device, but the error indicates it's on the GPU.
    # If `disp` is already a PyTorch tensor, you can skip `torch.tensor(disp)`.
    if not isinstance(disp, torch.Tensor):
        disp = torch.tensor(disp).float().cuda() # Assuming you need it on CUDA

    disp_perm = disp.permute(3, 0, 1, 2).unsqueeze(0)
    folding_ratio = []
    jac_det_n = dp.jacobian_det(disp_perm) # Assuming 'dp' is defined elsewhere
    
    # Calculate folding ratio and move to CPU before converting to numpy
    folding_tensor = (jac_det_n < 0).sum() / np.prod(jac_det_n.shape)
    folding_ratio.append(folding_tensor.cpu().numpy())

    # Calculate magnitude of gradient of Jacobian determinant
    mag_grad_jac_det = []
    # This line was already correctly using .cpu() in the original code, but
    # it's good to keep it consistent.
    mag_grad_jac_det.append(np.abs(np.gradient(jac_det_n[0][0].cpu().numpy())).mean())
   
    return np.mean(folding_ratio), np.mean(mag_grad_jac_det)