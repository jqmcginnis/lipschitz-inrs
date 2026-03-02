import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
import json
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
from lipschitz_budget_last import get_allocation_function

from geomloss import SamplesLoss

from models_constrained import ReluMLP, GroupSortMLP, HouseholderMLP
from pytorch3d.loss import chamfer_distance

def set_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_list(param_str, expected_length):
    """
    Parse parameter strings from command line that can be single values or comma-separated lists.
    """
    if ',' in param_str:
        param_values = [float(x.strip()) for x in param_str.split(',')]
        if len(param_values) != expected_length:
            raise ValueError(f"Number of values ({len(param_values)}) must match "
                           f"expected length ({expected_length})")
        return param_values
    else:
        param_value = float(param_str)
        return [param_value] * expected_length

def eikonal_loss(gradients: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.norm(gradients, dim=-1) - 1.0) ** 2)

def compute_gradients(network: nn.Module, points: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    points = points.clone().detach().requires_grad_(True)
    sdf_vals = network(points)
    grads = torch.autograd.grad(
        outputs=sdf_vals,
        inputs=points,
        grad_outputs=torch.ones_like(sdf_vals),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True
    )[0]
    return grads

def estimate_normals_pca(points: torch.Tensor, k: int = 30, batch_size: int = 1000) -> torch.Tensor:
    import torch.nn.functional as F
    from scipy.spatial import cKDTree

    device = points.device
    points_np = points.cpu().numpy()
    N = points.shape[0]
    normals = torch.zeros_like(points)

    tree = cKDTree(points_np)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_points = points[batch_start:batch_end]
        batch_points_np = batch_points.cpu().numpy()

        _, neighbors_idx = tree.query(batch_points_np, k=k)
        neighbors_pts = points[neighbors_idx]
        neighbors_mean = neighbors_pts.mean(dim=1, keepdim=True)
        centered = neighbors_pts - neighbors_mean
        cov_matrices = torch.matmul(centered.transpose(1, 2), centered) / (k - 1)
        _, eigenvecs = torch.linalg.eigh(cov_matrices)
        normals_batch = eigenvecs[:, :, 0]
        vectors_to_centroid = batch_points - points.mean(dim=0, keepdim=True)
        dot = (normals_batch * vectors_to_centroid).sum(dim=1, keepdim=True)
        normals_batch = torch.where(dot < 0, -normals_batch, normals_batch)

        normals[batch_start:batch_end] = normals_batch

    return F.normalize(normals, dim=-1)

def load_and_normalize_mesh(filepath: str, num_samples: int = 100000):
    """
    Loads a mesh, samples points uniformly from triangles, and returns points and outward normals.
    Handles sharp edges and thin features correctly.

    Returns:
        points (np.ndarray): shape (num_samples, 3)
        normals (np.ndarray): shape (num_samples, 3), outward-pointing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading mesh from {filepath}...")
    mesh = trimesh.load(filepath, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded file is not a valid mesh")

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight. Signed distance may be approximate.")

    # Sample points on the surface
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_indices].copy()

    # Compute signed distance for each point
    # Positive: outside, Negative: inside
    # Trimesh requires a watertight mesh; for non-watertight, result is approximate
    signed_dist = trimesh.proximity.signed_distance(mesh, points)

    # Flip normals if they point inward (toward negative signed distance)
    normals[signed_dist < 0] *= -1

    # Normalize points
    points_mean = points.mean(axis=0)
    points -= points_mean
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-8:
        points /= scale

    # Normalize normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    # Apply same normalization to the mesh
    mesh_normalized = mesh.copy()
    mesh_normalized.vertices -= points_mean
    mesh_normalized.vertices /= scale

    return points.astype(np.float32), normals.astype(np.float32), mesh_normalized

def calculate_and_save_metrics(
    gt_points: np.ndarray,
    rendered_mesh: trimesh.Trimesh,
    output_dir: str,
    epoch: int,
    mesh_freq: int, # Add this argument
    num_samples_cd: int = 30000,
    num_samples_emd: int = 500):

    """Calculates Chamfer Distance (CD) and Earth Mover's Distance (EMD) between GT points and rendered mesh using PyTorch3D and GeomLoss."""
    
    # we base the number of samples on the settings used in Park et al.'s DeepSDF paper
    # c.f. Table 2 https://arxiv.org/pdf/1901.05103

    # Check if CUDA is available
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') # pytorch3d not compiled with cuda thats why 

    # --- Chamfer Distance Calculation (30,000 points) ---
    print("Calculating Chamfer Distance with 30,000 points...")
    
    # Sample 30,000 points from the rendered mesh for CD
    rendered_points_cd_np, _ = trimesh.sample.sample_surface_even(rendered_mesh, num_samples_cd)
    
    # Convert GT and rendered points to PyTorch tensors and move to device
    gt_points_tensor_cd = torch.from_numpy(gt_points).float().to(device)
    rendered_points_tensor_cd = torch.from_numpy(rendered_points_cd_np).float().to(device)
    
    # Add a batch dimension for PyTorch3D
    gt_points_tensor_cd_batched = gt_points_tensor_cd.unsqueeze(0)
    rendered_points_tensor_cd_batched = rendered_points_tensor_cd.unsqueeze(0)

    # Calculate Chamfer Distance using PyTorch3D
    cd_loss_p1, cd_loss_p2 = chamfer_distance(rendered_points_tensor_cd_batched, gt_points_tensor_cd_batched)
    
    # Check if the outputs are valid
    cd = 0.0
    if cd_loss_p1 is not None:
        cd += cd_loss_p1.item()
    if cd_loss_p2 is not None:
        cd += cd_loss_p2.item()

    print(f"Chamfer Distance (PyTorch3D): {cd:.6f}")

    # --- Earth Mover's Distance Calculation (500 points) ---
    print("\nCalculating Earth Mover's Distance with 500 points...")

    # Sample 500 points from the rendered mesh for EMD
    rendered_points_emd_np, _ = trimesh.sample.sample_surface_even(rendered_mesh, num_samples_emd)

    # Convert GT and rendered points to PyTorch tensors and move to device
    gt_points_tensor_emd = torch.from_numpy(gt_points).float().to(device)
    rendered_points_tensor_emd = torch.from_numpy(rendered_points_emd_np).float().to(device)

    # EMD using SamplesLoss from GeomLoss (Sinkhorn approximation)
    emd_loss = SamplesLoss(loss='sinkhorn', p=2, blur=0.05)  # p=2 for Wasserstein-2 distance
    emd = emd_loss(gt_points_tensor_emd, rendered_points_tensor_emd)
    print(f"Earth Mover's Distance (Sinkhorn approx): {emd.item():.6f}")

    # --- Save Metrics ---
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    
    # Check if the file exists to decide if a header is needed
    write_header = not os.path.exists(metrics_path)

    with open(metrics_path, 'a') as f:
        if write_header:
            f.write("epoch,chamfer_distance,emd_distance\n")
        f.write(f"{epoch},{cd:.6f},{emd.item():.6f}\n")

    return cd, emd.item()

class PointCloudDataset(Dataset):
    def __init__(self,
                 surface_points: np.ndarray,
                 surface_normals: Optional[np.ndarray] = None,
                 num_samples: int = 100000,
                 coarse_scale: float = 0.05,
                 fine_scale: float = 0.005):
        super().__init__()
        self.surface_points = torch.from_numpy(surface_points).float()
        self.surface_normals = torch.from_numpy(surface_normals).float() if surface_normals is not None else None
        self.num_samples = num_samples
        self.coarse_scale = float(coarse_scale)
        self.fine_scale = float(fine_scale)

        self.kdtree = cKDTree(self.surface_points.numpy())
        self.use_kdtree = True
    
    def __len__(self):
        return int(self.num_samples)

    def _nearest_distance_and_index(self, pts: np.ndarray):
        if self.use_kdtree:
            dists, idxs = self.kdtree.query(pts, k=1)
            return dists, idxs

        pts_t = torch.from_numpy(pts).float()
        dists = torch.cdist(pts_t.unsqueeze(0), self.surface_points.unsqueeze(0))[0]
        min_vals, min_idx = dists.min(dim=1)
        return min_vals.numpy(), min_idx.numpy()

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.surface_points))
        surf_p = self.surface_points[i].numpy()
        
        scale = self.coarse_scale if np.random.rand() < 0.5 else self.fine_scale
        
        sample = surf_p + np.random.laplace(loc=0.0, scale=scale, size=3)
        sample = sample.astype(np.float32)

        dists, idxs = self._nearest_distance_and_index(sample.reshape(1, 3))
        unsigned_d = float(dists[0])
        nearest_idx = int(idxs[0])

        sign = 1.0
        surf_point = self.surface_points[nearest_idx]
        surf_normal = None
        if self.surface_normals is not None:
            surf_normal = self.surface_normals[nearest_idx]
            n = surf_normal.numpy()
            v = sample - surf_point.numpy()
            sign = 1.0 if np.dot(v, n) > 0 else -1.0
        
        sdf_target = unsigned_d * sign
        
        return torch.from_numpy(sample).float(), torch.tensor([sdf_target], dtype=torch.float32), surf_point, surf_normal

class SDFTrainer:
    def __init__(self,
                 network: GroupSortMLP,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 epochs: int = 100,
                 log_dir: Optional[str] = None,
                 checkpoint_freq: int = 100,
                 mesh_freq: int = 100,
                 resolution: int = 256):
        self.mesh_freq = mesh_freq
        self.resolution = resolution
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, amsgrad=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        self.losses = {'total': [], 'sdf': [], 'eikonal': [], 'normal': []}
        self.checkpoint_freq = checkpoint_freq

        if log_dir is None:
            log_dir = f"runs/sdf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir

        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir)
            self.writer.add_text('Model/Architecture', str(network))
        else:
            self.writer = None

        self.global_step = 0

    def train_step(self,
                   samples: torch.Tensor,
                   target_sdf: torch.Tensor,
                   surface_points: torch.Tensor,
                   surface_normals: Optional[torch.Tensor] = None,
                   eikonal_weight: float = 0.1,
                   normal_weight: float = 0.01) -> dict:

        samples = samples.to(self.device)
        target_sdf = target_sdf.to(self.device).squeeze(-1)
        surface_points = surface_points.to(self.device)

        self.network.train()
        self.optimizer.zero_grad()

        predicted_sdf = self.network(samples).squeeze(-1)

        sdf_loss = torch.nn.functional.l1_loss(predicted_sdf, target_sdf)

        eikonal_sample_size = min(512, samples.shape[0])
        if eikonal_sample_size > 0:
            idxs = torch.randperm(samples.shape[0])[:eikonal_sample_size]
            eikonal_samples = samples[idxs]
            grads = compute_gradients(self.network, eikonal_samples, create_graph=True)
            eik_loss_val = eikonal_loss(grads)
        else:
            eik_loss_val = torch.tensor(0.0, device=self.device)

        normal_loss_val = torch.tensor(0.0, device=self.device)
        if surface_normals is not None:
            surf_pts = surface_points.to(self.device)
            surf_grads = compute_gradients(self.network, surf_pts, create_graph=True)
            surf_grads_norm = torch.nn.functional.normalize(surf_grads, dim=-1)
            surface_normals = surface_normals.to(self.device)
            normal_loss_val = 1.0 - torch.mean(torch.abs(torch.sum(surf_grads_norm * surface_normals, dim=-1)))

        total_loss = sdf_loss + eikonal_weight * eik_loss_val + normal_weight * normal_loss_val

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'sdf': sdf_loss.item(),
            'eikonal': eik_loss_val.item() if isinstance(eik_loss_val, torch.Tensor) else float(eik_loss_val),
            'normal': normal_loss_val.item()
        }

    def train(self,
              dataloader: DataLoader,
              num_epochs: int = 100,
              eikonal_weight: float = 0.1,
              normal_weight: float = 0.01,
              estimate_normals: bool = True,
              print_freq: int = 10,
              vis_freq: int = 1):

        self.network.train()

        self.writer.add_text('Hyperparameters/eikonal_weight', str(eikonal_weight))
        self.writer.add_text('Hyperparameters/normal_weight', str(normal_weight))
        self.writer.add_text('Hyperparameters/learning_rate', str(self.optimizer.param_groups[0]['lr']))

        if estimate_normals and dataloader.dataset.surface_normals is None:
            print("Estimating surface normals...")
            start_time = time.time()
            surface_normals = estimate_normals_pca(dataloader.dataset.surface_points.to(self.device))
            dataloader.dataset.surface_normals = surface_normals.cpu()
            print(f"Normal estimation took {time.time() - start_time:.2f} seconds")

        for epoch in range(num_epochs):
            epoch_losses = {'total': 0, 'sdf': 0, 'eikonal': 0, 'normal': 0}
            epoch_start_time = time.time()

            for _, (samples, target_distances, surface_points, surface_normals) in enumerate(dataloader):
                losses = self.train_step(
                    samples,
                    target_distances,
                    surface_points,
                    surface_normals,
                    eikonal_weight,
                    normal_weight
                )

                for key in epoch_losses:
                    epoch_losses[key] += losses[key]

                self.writer.add_scalar('Loss/Batch/Total', losses['total'], self.global_step)
                self.writer.add_scalar('Loss/Batch/SDF', losses['sdf'], self.global_step)
                self.writer.add_scalar('Loss/Batch/Eikonal', losses['eikonal'], self.global_step)
                self.writer.add_scalar('Loss/Batch/Normal', losses['normal'], self.global_step)

                self.global_step += 1

            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader)
                self.losses[key].append(epoch_losses[key])

            self.writer.add_scalar('Loss/Epoch/Total', epoch_losses['total'], epoch)
            self.writer.add_scalar('Loss/Epoch/SDF', epoch_losses['sdf'], epoch)
            self.writer.add_scalar('Loss/Epoch/Eikonal', epoch_losses['eikonal'], epoch)
            self.writer.add_scalar('Loss/Epoch/Normal', epoch_losses['normal'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            self.writer.add_scalar('Time/Epoch_Duration', epoch_time, epoch)

            if epoch % print_freq == 0:
                print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.2f}s)")
                print(f"  Total Loss: {epoch_losses['total']:.6f}")
                print(f"  SDF Loss: {epoch_losses['sdf']:.6f}")
                print(f"  Eikonal Loss: {epoch_losses['eikonal']:.6f}")
                print(f"  Normal Loss: {epoch_losses['normal']:.6f}")
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if epoch % vis_freq == 0 and epoch > 0:
                self.visualize_sdf_slice(epoch)

            if epoch % self.mesh_freq == 0 and epoch > 0:
                print(f"[{datetime.now()}] Epoch {epoch}: Extracting and saving mesh...")
                
                # Extract the mesh from the trained SDF
                rendered_mesh = self.extract_mesh(resolution=self.resolution)
                
                # Only save if a valid mesh is extracted
                if rendered_mesh is not None and len(rendered_mesh.vertices) > 0:
                    mesh_filename = f"epoch_{epoch}_rendered_mesh.ply"
                    mesh_output_path = os.path.join(self.log_dir, mesh_filename)
                    rendered_mesh.export(mesh_output_path)
                    print(f"Saved rendered mesh to {mesh_output_path}")
                else:
                    print(f"Could not extract a valid mesh at epoch {epoch}. Skipping save.")

                if epoch % self.checkpoint_freq == 0 and epoch > 0:
                    self.save_checkpoint(epoch)

        self.save_checkpoint(num_epochs)
        self.writer.close()
        print(f"Training completed! Logs saved to {self.log_dir}")

    def visualize_sdf_slice(self, epoch: int, resolution: int = 256):
        """Visualize SDF values on XY, XZ, YZ slices with negative=blue, positive=red and log to TensorBoard"""
        self.network.eval()

        with torch.no_grad():
            coords = torch.linspace(-1, 1, resolution)
            xx, yy = torch.meshgrid(coords, coords, indexing='xy')
            zz = torch.zeros_like(xx)
            grid_xy = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).to(self.device)

            xx, zz = torch.meshgrid(coords, coords, indexing='xy')
            yy = torch.zeros_like(xx)
            grid_xz = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).to(self.device)

            yy, zz = torch.meshgrid(coords, coords, indexing='xy')
            xx = torch.zeros_like(yy)
            grid_yz = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).to(self.device)

            sdf_xy = self.network(grid_xy).squeeze().cpu().reshape(resolution, resolution)
            sdf_xz = self.network(grid_xz).squeeze().cpu().reshape(resolution, resolution)
            sdf_yz = self.network(grid_yz).squeeze().cpu().reshape(resolution, resolution)

            max_abs_sdf = torch.max(torch.tensor([sdf_xy.abs().max(), sdf_xz.abs().max(), sdf_yz.abs().max()]))
            sdf_xy_norm = sdf_xy / (max_abs_sdf + 1e-8)
            sdf_xz_norm = sdf_xz / (max_abs_sdf + 1e-8)
            sdf_yz_norm = sdf_yz / (max_abs_sdf + 1e-8)

            cmap = plt.get_cmap('bwr')

            def sdf_to_rgb(sdf):
                rgb = cmap((sdf.numpy() + 1) / 2.0)[:, :, :3]
                return torch.tensor(rgb).permute(2, 0, 1)

            sdf_xy_rgb = sdf_to_rgb(sdf_xy_norm)
            sdf_xz_rgb = sdf_to_rgb(sdf_xz_norm)
            sdf_yz_rgb = sdf_to_rgb(sdf_yz_norm)

            self.writer.add_image('SDF/XY_Slice', sdf_xy_rgb, epoch)
            self.writer.add_image('SDF/XZ_Slice', sdf_xz_rgb, epoch)
            self.writer.add_image('SDF/YZ_Slice', sdf_yz_rgb, epoch)

        self.network.train()

    def save_checkpoint(self, epoch: int):
        checkpoint_path = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': self.losses,
            'global_step': self.global_step
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.losses = checkpoint['losses']
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']

    def extract_mesh(self, resolution: int = 128, threshold: float = 0.0):
        self.network.eval()
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        z = np.linspace(-1, 1, resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        grid_points_tensor = torch.from_numpy(grid_points).float().to(self.device)

        with torch.no_grad():
            sdf_values = []
            batch_size = 100000
            for i in range(0, len(grid_points_tensor), batch_size):
                batch = grid_points_tensor[i:i + batch_size]
                sdf_batch = self.network(batch).squeeze()
                sdf_values.append(sdf_batch)

            sdf_values = torch.cat(sdf_values).cpu().numpy()

        sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
        try:
            verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=threshold)

            # Denormalize vertices to the original [-1, 1] bounding box
            # This logic only runs if marching_cubes is successful
            verts = verts / (resolution - 1) * 2.0 - 1.0

            return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
        except ValueError:
            print("Warning: Could not extract a mesh. The surface level may not be within the SDF value range.")
            return None


def main():
    parser = argparse.ArgumentParser(description="Train a Neural SDF from a point cloud.")
    
    parser.add_argument('--datafile', type=str, default="stanford-bunny.obj", help="Path to the input .ply or .stl file.")
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--log_freq', default=20, type=int)
    parser.add_argument('--vis_freq', default=100, type=int)
    parser.add_argument('--mesh_freq', type=int, default=100, help='Frequency in epochs to extract and save the mesh.')

    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--num_samples', default=100000, type=int)
    parser.add_argument('--k_fourier', default=1.0, type=float)
    parser.add_argument('--k_layers', default="1.0", type=str,
                        help="Lipschitz Layers - single value (e.g., '1.0') or comma-separated list.")
    parser.add_argument('--k_acts', default="1.0", type=str,
                        help="Lipschitz for activation - single value (e.g., '30.0') or comma-separated list.")
    parser.add_argument('--allocation', choices=['uniform', 'linear', 'exponential', 'cosine', 'midheavy', 'first'], default='exponential')
    parser.add_argument('--r_budget', default=1.0, type=float)
    parser.add_argument('--norm_type', choices=['spectral_norm', 'bjoerck', 'sll'], default='sll', type=str)
    parser.add_argument('--num_layers', default=8, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--model', choices=['relu_mlp', 'groupsort_mlp', 'householder_mlp'], default='groupsort_mlp')
    parser.add_argument('--seed', default=12, type=int, help='Random seed for reproducibility')

    parser.add_argument('--batch_size', type=int, default=16384, help='Number of samples per batch.')
    parser.add_argument('--group_size', type=int, default=2, help='GroupSort Group Size.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate.')

    parser.add_argument('--eikonal_weight', type=float, default=0.01,
                        help='Weight for the eikonal loss.')
    parser.add_argument('--normal_weight', type=float, default=0.01,
                        help='Weight for the normal loss (if normals are available).')
    parser.add_argument('--data_root', type=str, default='../data/common-3d-test-models/', help='Directory containing input data files.')
    parser.add_argument('--log_dir', type=str, default='../logs/sdf/', help='Directory for TensorBoard logs and checkpoints.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training (e.g., "cuda" or "cpu").')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers.')
    args = parser.parse_args()

    set_seed(args.seed)
    allocation_fn = get_allocation_function(args.allocation)
    k_allocation = allocation_fn(1.0, args.num_layers, r=args.r_budget)[0]
    args.k_layers = list(k_allocation) #parse_list(args.k_layers, args.num_layers)      
    args.k_acts = parse_list(args.k_acts, args.num_layers - 1)

    k_layers_str = "-".join(map(str, args.k_layers))
    k_acts_str = "-".join(map(str, args.k_acts))

    # Create a unique, descriptive directory for this run's outputs
    output_dir_name = f"{args.model}_e{args.epochs}_lr{args.learning_rate}_kl{k_layers_str}_ka{k_acts_str}_kf{args.k_fourier}_n{args.norm_type}"
    dataset_name = os.path.splitext(os.path.basename(args.datafile))[0].replace(".obj","").replace(".ply","")
    output_dir = os.path.join(args.log_dir, f"{dataset_name}_{output_dir_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {output_dir}")

    # Save all arguments and settings to a JSON file
    settings_path = os.path.join(output_dir, "settings.json")
    with open(settings_path, 'w') as f:
        settings = vars(args)
        # Ensure lists are serialized correctly
        settings['k_layers'] = args.k_layers
        settings['k_acts'] = args.k_acts
        json.dump(settings, f, indent=4)
    print(f"Run settings saved to {settings_path}")

    # Load and normalize the mesh
    # normalized_points, normalized_normals, scale, centroid = load_and_normalize_mesh(args.datafile)
    # normalized_points, normalized_normals= load_and_normalize_mesh(args.datafile, args.num_samples)
    normalized_points, normalized_normals, normalized_mesh = load_and_normalize_mesh(args.data_root+args.datafile, args.num_samples)

    print(f"Loaded {len(normalized_points)} surface points")
    print(f"Point cloud bounds: min={normalized_points.min(axis=0)}, max={normalized_points.max(axis=0)}")

    # Create dataset and dataloader
    dataset = PointCloudDataset(
        surface_points=normalized_points,
        surface_normals=normalized_normals,
        num_samples=args.num_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    if args.model == 'relu_mlp':
        model = ReluMLP(input_dim=3, hidden_dim=args.hidden_dim,
                        output_dim=1, num_layers=args.num_layers, k_layers=args.k_layers, norm_type=args.norm_type)
    elif args.model == 'groupsort_mlp':
        model = GroupSortMLP(input_dim=3, hidden_dim=args.hidden_dim,
                        output_dim=1, num_layers=args.num_layers, k_layers=args.k_layers, norm_type=args.norm_type, group_size=args.group_size)  
    elif args.model == 'householder_mlp':
        model = HouseholderMLP(input_dim=3, hidden_dim=args.hidden_dim,
                        output_dim=1, num_layers=args.num_layers, k_layers=args.k_layers, norm_type=args.norm_type)  
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Using device: {args.device}")

    trainer = SDFTrainer(
        network=model,
        device=args.device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        checkpoint_freq=args.vis_freq,
        mesh_freq=args.mesh_freq, 
        resolution=args.resolution, 
        log_dir=output_dir
    )

    print("Starting training...")
    trainer.train(
        dataloader=dataloader,
        num_epochs=args.epochs,
        eikonal_weight=args.eikonal_weight,
        normal_weight=args.normal_weight,
        estimate_normals=True,
        print_freq=args.log_freq,
        vis_freq=args.vis_freq,
    )

    print("Training completed!")

    # Base filename for outputs
    filename_base = dataset_name
    
    # 1. Save normalized GT mesh
    gt_mesh_output_path = os.path.join(output_dir, f"{filename_base}_normalized_gt_mesh.ply")
    normalized_mesh.export(gt_mesh_output_path)
    print(f"Saved normalized GT mesh to {gt_mesh_output_path}")

    # 2. Save normalized GT point cloud
    gt_pcd = trimesh.points.PointCloud(normalized_points)
    gt_output_path = os.path.join(output_dir, f"{filename_base}_normalized_gt_points.ply")
    gt_pcd.export(gt_output_path)
    print(f"Saved normalized GT point cloud to {gt_output_path}")

    # 3. Save PyTorch model
    model_output_path = os.path.join(output_dir, f"{filename_base}_model.pt")
    torch.save(model.state_dict(), model_output_path)
    print(f"Saved trained PyTorch model to {model_output_path}")

    # 4. Extract and save final rendered mesh
    print(f"[{datetime.now()}] Training finished. Extracting final mesh...")
    final_rendered_mesh = trainer.extract_mesh(resolution=args.resolution)
    final_mesh_output_path = os.path.join(output_dir, f"{filename_base}_final_mesh.ply")
    
    if final_rendered_mesh is not None and len(final_rendered_mesh.vertices) > 0:
        print(f"Extracted final mesh with {len(final_rendered_mesh.vertices)} vertices and {len(final_rendered_mesh.faces)} faces")
        final_rendered_mesh.export(final_mesh_output_path)
        print(f"Saved final rendered mesh to {final_mesh_output_path}")

        # 5. Calculate and save metrics for the final mesh
        print("Calculating and saving final metrics...")
        calculate_and_save_metrics(normalized_points, final_rendered_mesh, output_dir, epoch=args.epochs, mesh_freq=args.mesh_freq)
    else:
        print("Could not extract a valid final mesh. Skipping final metric calculation.")

if __name__ == "__main__":
    main()
