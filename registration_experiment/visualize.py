import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from vis_utils import visualise_result

def load_nifti_as_array(filepath):
    """Load NIfTI file and return as numpy array"""
    sitk_img = sitk.ReadImage(filepath)
    return sitk.GetArrayFromImage(sitk_img)

def load_and_visualize_idir_results(output_dir, case_id, slice_axis=0, save_vis=True, scale_factor_def=1, metrics = {}):
    """
    Load IDIR results and create visualizations
    
    Args:
        output_dir: Directory containing the results
        case_id: Case number
        slice_axis: Which axis to slice for 3D visualization (0, 1, or 2)
        save_vis: Whether to save visualization figures
    """
    
    # Load all the files
    fixed_path = os.path.join(output_dir, f"case{case_id}_fixed.nii.gz")
    moving_path = os.path.join(output_dir, f"case{case_id}_moving.nii.gz")
    warped_path = os.path.join(output_dir, f"case{case_id}_transformed_moving.nii.gz")
    deform_path = os.path.join(output_dir, f"case{case_id}_deformation_field.nii.gz")
    mask_path = os.path.join(output_dir, f"case{case_id}_mask.nii.gz")
    
    fixed_img = load_nifti_as_array(fixed_path)
    moving_img = load_nifti_as_array(moving_path)
    warped_img = load_nifti_as_array(warped_path)
    deform_field = load_nifti_as_array(deform_path)
    mask_img = load_nifti_as_array(mask_path)
   
    # Convert to torch tensors for compatibility with your vis functions
    fixed_torch = torch.from_numpy(fixed_img).float()
    moving_torch = torch.from_numpy(moving_img).float()
    warped_torch = torch.from_numpy(warped_img).float()
    mask_img_torch = torch.from_numpy(mask_img).float()
    
    # Deformation field needs to be rearranged: (Z,Y,X,3) -> (3,Z,Y,X)
    if deform_field.ndim == 4 and deform_field.shape[-1] == 3:
        deform_field = np.transpose(deform_field, (3, 0, 1, 2))
    deform_torch = torch.from_numpy(deform_field).float()

    # Add batch and channel dimensions: (1, 1, Z, Y, X)
    data_dict = {
        'fixed': fixed_torch.unsqueeze(0).unsqueeze(0),
        'moving': moving_torch.unsqueeze(0).unsqueeze(0),
        'warped': warped_torch.unsqueeze(0).unsqueeze(0),
        'mask': mask_img_torch.unsqueeze(0).unsqueeze(0),
        'disp_pred': scale_factor_def * deform_torch.unsqueeze(0),  # (1, 3, Z, Y, X)
    }
    
    fig2 = visualise_result(
        data_dict=data_dict,
        metrics_dict=metrics,
        axis=slice_axis,
        save_result_dir=output_dir if save_vis else None,
        epoch=0,  # Just use 0 since this is final result
        id=case_id,
        show=False,
        close=False,

    )
       
    return fig2