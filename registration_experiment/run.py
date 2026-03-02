import argparse
from utils import general
from models import models
import torch
import SimpleITK as sitk
import os
import numpy as np
import torch.nn.functional as F
from visualize import load_and_visualize_idir_results
from log_results import save_metrics_to_csv
from objectives.ncc import NCC
from utils.custom_utils import *
from lipschitz_budget_last import get_allocation_function

parser = argparse.ArgumentParser(description="Run implicit registration.")
parser.add_argument('--norm_type', choices=['spectral_norm', 'bjoerck', 'sll'], default='spectral_norm', type=str)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)

parser.add_argument('--alpha_bending', default=100.0, type=float) # TODO: either 100 or 1000
parser.add_argument('--hyper_regularization', type=bool, default=True, 
                    help='Enable hyper-regularization.')
parser.add_argument('--jacobian_regularization', type=bool, default=True, 
                    help='Enable Jacobian regularization.')
parser.add_argument('--bending_regularization', type=bool, default=True, 
                    help='Enable bending regularization.')
# DONT CHANGE VAR names because of kwargs of original codebase
parser.add_argument('--network_type', type=str, default='Lipschitz_Siren', 
                    choices=['MLP', 'Siren', 'Vanilla_FFN', 'Lipschitz_Siren', 'Lipschitz_FFN', ], help='Type of network to use.')
parser.add_argument('--epochs', type=int, default=20, 
                    help='Number of training epochs.')
parser.add_argument('--omega', type=float, default=30.0, 
                    help='Value for the omega parameter in SIREN.')
parser.add_argument('--case_id', type=int, default=10, # TODO: corresponds to differnet images
                    help='Case ID for DIRLAB dataset.')
parser.add_argument('--data_dir', type=str, 
                    default='../data/IDIR', # TODO: corresponds to differnet images
                    help='Output directory for results.')
parser.add_argument('--out_dir', type=str, 
                    default='../logs/DIRLAB', # TODO: corresponds to differnet images
                    help='Output directory for results.')
parser.add_argument('--allocation', choices=['uniform', 'linear', 'exponential', 'cosine', 'midheavy', 'first'], default='exponential')
parser.add_argument('--r_budget', default=1.0, type=float)

args = parser.parse_args()
case_id = args.case_id
allocation_fn = get_allocation_function(args.allocation)  # Validate allocation method
if args.network_type == "Lipschitz_Siren":
    k_allocation = allocation_fn(2.0, 2*args.num_layers-1, r=args.r_budget)[0]
    k_layers = list(k_allocation[0::2])
    k_acts = list(k_allocation[1::2])
    k_fourier = 1.0
    out_dir = os.path.join(args.out_dir, f"case_{case_id}_{args.network_type}_{args.epochs}_{args.allocation}_{args.r_budget}_{args.num_layers}_{args.norm_type}_{args.hidden_dim}")
elif args.network_type == "Lipschitz_FFN":
    k_allocation = allocation_fn(2.0, args.num_layers+1, r=args.r_budget)[0]
    k_fourier = k_allocation[0]
    k_layers = list(k_allocation[1:])
    k_acts = [1.0]*(args.num_layers-1)
    out_dir = os.path.join(args.out_dir, f"case_{case_id}_{args.network_type}_{args.epochs}_{args.allocation}_{args.r_budget}_{args.num_layers}_{args.norm_type}_{args.hidden_dim}")
elif args.network_type == "Vanilla_FFN":
    out_dir = os.path.join(args.out_dir, f"case_{case_id}_{args.network_type}_{args.epochs}_{args.k_fourier}_{args.num_layers}_{args.hidden_dim}")
elif args.network_type == "MLP":
    out_dir = os.path.join(args.out_dir, f"case_{case_id}_{args.network_type}_{args.epochs}_{args.num_layers}_{args.hidden_dim}")
else:
    out_dir = os.path.join(args.out_dir, f"case_{case_id}_{args.network_type}_{args.epochs}_{args.omega}_{args.num_layers}_{args.hidden_dim}")

os.makedirs(out_dir,exist_ok=True)
(
    img_insp,
    img_exp,
    landmarks_insp,
    landmarks_exp,
    mask_exp,
    voxel_size,
) = general.load_image_DIRLab(case_id, "{}/Case".format(args.data_dir), out_dir)

# Build the kwargs dictionary and update other variables
kwargs = {
    'verbose': False, 
    'hyper_regularization': args.hyper_regularization,
    'jacobian_regularization': args.jacobian_regularization,
    'bending_regularization': args.bending_regularization,
    'network_type': args.network_type,
    'epochs': args.epochs,
    'omega': args.omega,
    'mask': mask_exp,
    'save_folder': out_dir,
    'k_layers': k_layers,
    'k_act': k_acts,
    'k_fourier': k_fourier,
    'norm_type': args.norm_type,
    'num_layers': args.num_layers,
    'hidden_dim': args.hidden_dim,
    'alpha_beinding': args.alpha_bending

}

ImpReg = models.ImplicitRegistrator(img_exp, img_insp, **kwargs)
ImpReg.fit()
new_landmarks_orig, _ = general.compute_landmarks(
    ImpReg.network, landmarks_insp, image_size=img_insp.shape
)

np.savetxt(f"{out_dir}/case{case_id}_transformed_landmarks.txt", 
           new_landmarks_orig, 
           fmt='%d', 
           delimiter='\t')

metrics = {}

# DEFORMATION FIELD: Create coordinate tensor for the full image (and mask later)
coord_tensor = general.make_coordinate_tensor(img_insp.shape, gpu=True)
mask_exp = torch.tensor(mask_exp).bool()

with torch.no_grad():
    # Get the deformation field (displacement vectors)
    # deformation_field = ImpReg.network(coord_tensor)  # This is the displacement field
    # do batch processing if memory error
    deformation_field = torch.zeros(coord_tensor.shape, device=coord_tensor.device)
    batch_size = 100000  # Adjust based on your GPU memory
    for i in range(0, coord_tensor.shape[0], batch_size):
        end_idx = min(i + batch_size, coord_tensor.shape[0])
        deformation_field[i:end_idx] = ImpReg.network(coord_tensor[i:end_idx])
    mask_exp_def = mask_exp.clone().to(deformation_field.device)
    # no erosion
    # mask_exp_def = erode_mask_3d(mask_exp_def, erosion_size=0)  # Erode mask to avoid boundary artifacts

   # Apply mask - preserve identity (zero displacement) in masked regions
    identity_displacement = torch.zeros_like(deformation_field)
    masked_deformation_field = torch.where(
        mask_exp_def.flatten().unsqueeze(-1).expand_as(deformation_field),
        deformation_field,  # Keep original deformation where mask is True
        identity_displacement  # Zero displacement (identity) where mask is False
    )
    
    # Compute absolute coordinates for image transformation
    absolute_coords = torch.add(masked_deformation_field, coord_tensor)
    deformation_field_norm = coordinate_to_voxel_indices(masked_deformation_field.clone(), img_insp.shape)
    reshaped_field = deformation_field_norm.clone().reshape((*img_insp.shape, 3))
    folding_ratio, jacobian_det = calculate_jacobian_metrics(reshaped_field)
    
    print(f"Folding ratio: {folding_ratio:.4f} ({folding_ratio*100:.2f}%)")
    metrics["folding_ratio"] = folding_ratio
    
    # Transform the moving image
    transformed_image = ImpReg.transform_no_add(absolute_coords)
    transformed_image = transformed_image.view(img_insp.shape)
   
  

# Save transformed image
fixed_sitk_path = os.path.join(out_dir, f"case{case_id}_fixed.nii.gz")
fixed_sitk_img = sitk.ReadImage(fixed_sitk_path)

transformed_image_np = transformed_image.cpu().detach().numpy()
transformed_sitk = sitk.GetImageFromArray(transformed_image_np)
transformed_sitk.SetSpacing(fixed_sitk_img.GetSpacing())
transformed_sitk.SetOrigin(fixed_sitk_img.GetOrigin())
transformed_sitk.SetDirection(fixed_sitk_img.GetDirection())
sitk.WriteImage(transformed_sitk, f"{out_dir}/case{case_id}_transformed_moving.nii.gz")

# Save deformation field
deformation_field_and_coords_np = absolute_coords.cpu().detach().numpy()
# Reshape to (Z, Y, X, 3) format - 3 components for X, Y, Z displacements
deformation_coords_np = deformation_field_and_coords_np.reshape((*img_insp.shape, 3))
deformation_sitk = sitk.GetImageFromArray(deformation_coords_np, isVector=True)
deformation_sitk.SetSpacing(fixed_sitk_img.GetSpacing())
deformation_sitk.SetOrigin(fixed_sitk_img.GetOrigin())
deformation_sitk.SetDirection(fixed_sitk_img.GetDirection())
sitk.WriteImage(deformation_sitk, f"{out_dir}/case{case_id}_deformation_field_coords.nii.gz")

# Save deformation field
deformation_field_np = deformation_field.cpu().detach().numpy()
# Reshape to (Z, Y, X, 3) format - 3 components for X, Y, Z displacements
deformation_field_np = deformation_field_np.reshape((*img_insp.shape, 3))
deformation_sitk = sitk.GetImageFromArray(deformation_field_np, isVector=True)
deformation_sitk.SetSpacing(fixed_sitk_img.GetSpacing())
deformation_sitk.SetOrigin(fixed_sitk_img.GetOrigin())
deformation_sitk.SetDirection(fixed_sitk_img.GetDirection())
sitk.WriteImage(deformation_sitk, f"{out_dir}/case{case_id}_deformation_field.nii.gz")

print("TRE before:")
accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
    landmarks_insp, landmarks_exp, voxel_size=voxel_size
)
metrics["tre_mean_before"] = accuracy_mean[0]
metrics["tre_std_before"] = accuracy_std[0]

print("{} {} {}".format(case_id, accuracy_mean, accuracy_std))
print("TRE after:")
accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
    new_landmarks_orig, landmarks_exp, voxel_size=voxel_size
)
metrics["tre_mean_after"] = accuracy_mean[0]
metrics["tre_std_after"] = accuracy_std[0]
print("{} {} {}".format(case_id, accuracy_mean, accuracy_std))

ncc_metric = NCC()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create masked versions 
img_insp_masked = img_insp[mask_exp]
img_exp_masked = img_exp[mask_exp]
transformed_image_masked = transformed_image[mask_exp]

ncc_score = ncc_metric.ncc(img_insp_masked.to(device), img_exp_masked.to(device))
print("NCC before warp", ncc_score.cpu().item())
ncc_score = ncc_metric.ncc(transformed_image_masked.to(device), img_insp_masked.to(device))
print("NCC after warp", ncc_score.cpu().item())

# save visualization
load_and_visualize_idir_results(out_dir, case_id, slice_axis=0, save_vis=True, scale_factor_def=max(img_insp.shape), metrics=metrics)
load_and_visualize_idir_results(out_dir, case_id, slice_axis=1, save_vis=True, scale_factor_def=max(img_insp.shape), metrics=metrics)

save_metrics_to_csv(
    case_id,
    accuracy_mean_before=accuracy_mean,
    accuracy_std_before=accuracy_std,
    accuracy_mean_after=accuracy_mean,
    accuracy_std_after=accuracy_std,
    ncc_before=ncc_score.cpu().item(),
    ncc_after=ncc_score.cpu().item(),
    folding_ratio=folding_ratio,
    jacobian_det=jacobian_det,
    metrics=metrics,
    output_dir=os.path.join(out_dir,f"case_{case_id}_metrics.csv")
)

