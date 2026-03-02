import torch
import torch.nn.functional as F
import numpy as np

def compute_jacobian_determinant_3d(deformation_field):
    """
    Compute the Jacobian determinant of a 3D deformation field.
    
    Args:
        deformation_field: torch.Tensor of shape (D, H, W, 3) or (1, 3, D, H, W)
                          representing displacement vectors
    
    Returns:
        jacobian_det: torch.Tensor of shape (D, H, W) with Jacobian determinants
    """
    # Ensure the deformation field is in the right format (1, 3, D, H, W)
    if deformation_field.dim() == 4:  # (D, H, W, 3)
        deformation_field = deformation_field.permute(3, 0, 1, 2).unsqueeze(0)
    elif deformation_field.dim() == 5 and deformation_field.shape[0] != 1:
        deformation_field = deformation_field.squeeze()
    
    # Extract displacement components
    dx = deformation_field[0, 0, :, :, :]  # displacement in x
    dy = deformation_field[0, 1, :, :, :]  # displacement in y
    dz = deformation_field[0, 2, :, :, :]  # displacement in z
    
    # Compute gradients using finite differences
    # Gradients of dx
    dx_dx = torch.gradient(dx, dim=2)[0]  # ∂dx/∂x
    dx_dy = torch.gradient(dx, dim=1)[0]  # ∂dx/∂y
    dx_dz = torch.gradient(dx, dim=0)[0]  # ∂dx/∂z
    
    # Gradients of dy
    dy_dx = torch.gradient(dy, dim=2)[0]  # ∂dy/∂x
    dy_dy = torch.gradient(dy, dim=1)[0]  # ∂dy/∂y
    dy_dz = torch.gradient(dy, dim=0)[0]  # ∂dy/∂z
    
    # Gradients of dz
    dz_dx = torch.gradient(dz, dim=2)[0]  # ∂dz/∂x
    dz_dy = torch.gradient(dz, dim=1)[0]  # ∂dz/∂y
    dz_dz = torch.gradient(dz, dim=0)[0]  # ∂dz/∂z
    
    # Add identity matrix (since we have displacement field, not absolute coordinates)
    # Jacobian = I + ∇u where u is displacement
    dx_dx += 1.0
    dy_dy += 1.0
    dz_dz += 1.0
    
    # Compute 3x3 determinant
    # det = a11(a22*a33 - a23*a32) - a12(a21*a33 - a23*a31) + a13(a21*a32 - a22*a31)
    jacobian_det = (dx_dx * (dy_dy * dz_dz - dy_dz * dz_dy) - 
                    dx_dy * (dy_dx * dz_dz - dy_dz * dz_dx) + 
                    dx_dz * (dy_dx * dz_dy - dy_dy * dz_dx))
    
    return jacobian_det

def compute_folding_ratio(deformation_field, threshold=0.0):
    """
    Compute the folding ratio of a deformation field.
    
    Args:
        deformation_field: torch.Tensor representing displacement vectors
        threshold: float, threshold below which voxels are considered folded
    
    Returns:
        folding_ratio: float, ratio of folded voxels
        jacobian_det: torch.Tensor, Jacobian determinants for visualization
    """
    # Compute Jacobian determinant
    jacobian_det = compute_jacobian_determinant_3d(deformation_field)
    
    # Count folded voxels (negative or near-zero Jacobian determinant)
    folded_voxels = (jacobian_det <= threshold).sum().item()
    total_voxels = jacobian_det.numel()
    
    folding_ratio = folded_voxels / total_voxels
    
    return folding_ratio, jacobian_det
