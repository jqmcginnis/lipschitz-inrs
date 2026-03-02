import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from skimage.metrics import structural_similarity as ssim
from models_constrained import ReluFFN, ReluMLP, GaussFFN, GaussMLP, SirenMLP, GroupSortFFN, GroupSortMLP, PositionalEncodingReluMlp
from oracle import estimate_rgb_lipschitz_rectangular, visualize_rgb_lipschitz
import wandb
import lpips
import random
import os
from lipschitz_budget_last import get_allocation_function
from pathlib import Path

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

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(x) for x in obj]
    else:
        return obj

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)

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

def get_coordinates(h, w):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # Use 'xy' indexing
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return coords

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_lpips(img1, img2, lpips_model, device):
    """Compute LPIPS between two images"""
    # Convert images to tensors and normalize to [-1, 1] for LPIPS
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Rearrange from HWC to NCHW and normalize to [-1, 1]
    img1 = (img1.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).contiguous().to(device)
    img2 = (img2.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).contiguous().to(device)
    
    with torch.no_grad():
        try:
            old = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False   # avoid flaky conv algos
            val = lpips_model(img1.to(device), img2.to(device))
        except RuntimeError as e:
            # fallback if a GPU/cudnn kernel still blows up
            lpips_model_cpu = lpips_model.to('cpu')
            val = lpips_model_cpu(img1.to('cpu'), img2.to('cpu'))
        finally:
            torch.backends.cudnn.enabled = old

    return float(val)

def create_comparison_image(gt_img, recon_img, epoch, psnr_val, ssim_val, lpips_val):
    """Create side-by-side comparison of ground truth and reconstructed image"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    # Reconstruction
    axes[1].imshow(np.clip(recon_img, 0, 1))
    axes[1].set_title(f'Reconstruction\nEpoch {epoch}', fontsize=12)
    axes[1].axis('off')
    
    # Difference (absolute error amplified for visibility)
    diff = np.abs(gt_img - recon_img)
    diff_amplified = np.clip(diff * 5, 0, 1)  # Amplify difference for better visibility
    axes[2].imshow(diff_amplified)
    
    title = f'Difference (5x)\nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}\nLPIPS: {lpips_val:.4f}'
    
    axes[2].set_title(title, fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)  # Can also use 'vgg' or 'squeeze'
    
    # Load image
    img = load_image(args.data_root+args.image)
    h, w, c = img.shape
    
    # Flatten the image and coordinates
    coords = get_coordinates(h, w).to(device)
    target = img.view(-1, c).to(device)

    # Calculate the theoretical maximum Lipschitz constant based on resolution and color space
    min_input_dist = min(2 / (w - 1), 2 / (h - 1)) if w > 1 and h > 1 else 1e-6
    theoretical_max_lipschitz = np.sqrt(3) / min_input_dist
    wandb.log({"oracle/theoretical_max_lipschitz": theoretical_max_lipschitz})
    
    # --- INPAINTING CHANGES START HERE ---
    if args.inpainting_ratio < 1.0:
        # Create a mask for the pixels to train on
        num_pixels = h * w
        num_train_pixels = int(num_pixels * args.inpainting_ratio)
        
        # Get random indices without replacement
        indices = torch.randperm(num_pixels, device=device)[:num_train_pixels]
        
        # Create masked versions of coords and target for training
        masked_coords = coords[indices]
        masked_target = target[indices]
        
        # Create a copy of the original image with masked pixels for visualization
        masked_img = img.clone().view(-1, c)
        masked_img[~torch.isin(torch.arange(num_pixels), indices.cpu())] = 0.0 # set excluded pixels to black
        masked_img_display = masked_img.view(h, w, c).numpy()
        wandb.log({"images/inpainting_mask": wandb.Image(masked_img_display, caption=f'Inpainting Mask: {args.inpainting_ratio*100}% of pixels used for training')})
    else:
        masked_coords = coords
        masked_target = target
    # --- INPAINTING CHANGES END HERE ---
    
    # Lipschitz naive oracle estimate
    img_array = np.array(Image.open(args.data_root+args.image).convert('RGB'))
    lipschitz_estimate = estimate_rgb_lipschitz_rectangular(img_array, x_range=(-1, 1), y_range=(-1, 1))
    
    # --- NEW: Extract oracle values for logging ---
    oracle_lipschitz_max = lipschitz_estimate['combined']['lipschitz_max']
    oracle_lipschitz_mean = lipschitz_estimate['combined']['lipschitz_mean']
    oracle_lipschitz_spectral = lipschitz_estimate['combined']['lipschitz_spectral']
    print(f"Oracle Lipschitz Estimates - Max: {oracle_lipschitz_max:.2f}, Mean: {oracle_lipschitz_mean:.2f}, Spectral: {oracle_lipschitz_spectral:.2f}")
    wandb.log({
        "oracle/lipschitz_estimate_max": oracle_lipschitz_max,
        "oracle/lipschitz_estimate_mean": oracle_lipschitz_mean,
        "oracle/lipschitz_estimate_spectral": oracle_lipschitz_spectral,
    })
    
    _, image2 = visualize_rgb_lipschitz(img_array, lipschitz_estimate, x_range=(-1, 1), y_range=(-1, 1))
    wandb.log({"RGB Lipschitz Analysis": [wandb.Image(image2)]})

    allocation_fn = get_allocation_function(args.allocation)  # Validate allocation method
    k_budget = oracle_lipschitz_spectral+args.k_budget
    # Create model
    if args.model == 'relu_ffn':
        k_allocation = allocation_fn(k_budget, args.num_layers+1, r=args.r_budget)[0]
        k_fourier = k_allocation[0]
        k_layers = list(k_allocation[1:])
        model = ReluFFN(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                    output_dim=c, num_layers=args.num_layers, 
                    k_ff=k_fourier, k_layers=k_layers, norm_type=args.norm_type)  
    elif args.model == 'groupsort_ffn':
        k_allocation = allocation_fn(k_budget, args.num_layers+1, r=args.r_budget)[0]
        k_fourier = k_allocation[0]
        k_layers = list(k_allocation[1:])
        model = GroupSortFFN(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                    output_dim=c, num_layers=args.num_layers, 
                    k_ff=k_fourier, k_layers=k_layers, norm_type=args.norm_type) 
    elif args.model == 'groupsort_mlp':
        k_allocation = allocation_fn(k_budget, args.num_layers, r=args.r_budget)[0]
        k_layers = list(k_allocation)
        model = GroupSortMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, k_layers=k_layers, norm_type=args.norm_type) 
    elif args.model == 'relu_mlp':
        k_allocation = allocation_fn(k_budget, args.num_layers, r=args.r_budget)[0]
        k_layers = list(k_allocation)
        model = ReluMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, k_layers=k_layers, norm_type=args.norm_type)  
    elif args.model == 'relu_posencoding':
        k_allocation = allocation_fn(k_budget, args.num_layers+1, r=args.r_budget)[0]
        k_feat = k_allocation[0]
        k_layers = list(k_allocation[1:])
        model = PositionalEncodingReluMlp(input_dim=2, hidden_dim=args.hidden_dim, k_ff=k_feat,
                        output_dim=c, num_layers=args.num_layers, k_layers=k_layers, norm_type=args.norm_type)
    elif args.model == 'gauss_ffn':
        k_allocation = allocation_fn(k_budget, 2*args.num_layers, r=args.r_budget)[0]
        k_fourier = k_allocation[0]
        k_layers = list(k_allocation[1::2])
        k_acts = list(k_allocation[2::2])
        model = GaussFFN(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, 
                        k_ff=k_fourier, k_layers=k_layers, k_act=k_acts, norm_type=args.norm_type)  
    elif args.model == 'gauss_mlp':
        k_allocation = allocation_fn(k_budget, 2*args.num_layers-1, r=args.r_budget)[0]
        k_layers = list(k_allocation[0::2])
        k_acts =list( k_allocation[1::2])
        model = GaussMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, 
                        k_layers=k_layers, k_act=k_acts, norm_type=args.norm_type)  
    elif args.model == 'siren_mlp':
        k_allocation = allocation_fn(k_budget, 2*args.num_layers-1, r=args.r_budget)[0]
        k_layers = list(k_allocation[0::2])
        k_acts = list(k_allocation[1::2])
        model = SirenMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, 
                        k_layers=k_layers, k_act=k_acts, norm_type=args.norm_type)  
    elif args.model == 'wire_mlp':
        # Note: WireMLP appears to be missing from your constrained models file
        # You'll need to implement this or remove this option
        raise NotImplementedError("WireMLP not implemented in constrained models")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'epochs': []}
    layer_metrics = {}
    
    # Log original image to wandb
    gt_img_np = img.numpy()
    wandb.log({"images/ground_truth": wandb.Image(gt_img_np, caption="Ground Truth Image")})
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        # --- USE MASKED DATA FOR TRAINING ---
        pred = model(masked_coords)
        loss = F.mse_loss(pred, masked_target)
        # --- END OF MASKED DATA USE ---

        loss.backward()
        optimizer.step()
        
        if epoch % args.log_n_epochs == 0:
            with torch.no_grad():
                # For logging and evaluation, use ALL coordinates (unmasked)
                full_pred = model(coords).view(h, w, c).cpu().numpy()
                target_img = img.numpy()
                
                psnr_val = psnr(torch.tensor(full_pred), torch.tensor(target_img)).item()
                ssim_val = ssim(target_img, full_pred, channel_axis=-1, data_range=1.0)
                
                # Compute LPIPS
                lpips_val = compute_lpips(target_img, full_pred, lpips_model, device)
                
                metrics['psnr'].append(psnr_val)
                metrics['ssim'].append(ssim_val)
                metrics['lpips'].append(lpips_val)
                metrics['epochs'].append(epoch)
                
                # Print logging
                print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                
                # Create log dict
                log_dict = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'lpips': lpips_val
                }
                
                # Only log image evolution if requested
                if args.log_image_evolution:
                    # Create comparison image
                    comparison_fig = create_comparison_image(target_img, full_pred, epoch, psnr_val, ssim_val, lpips_val)
                    
                    log_dict.update({
                        'images/comparison': wandb.Image(comparison_fig, caption=f'Epoch {epoch}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}'),
                        'images/reconstruction': wandb.Image(np.clip(full_pred, 0, 1), caption=f'Reconstruction at epoch {epoch}')
                    })
                    
                    plt.close(comparison_fig)  # Close figure to free memory
                
                # Log layer metrics
                if hasattr(model, 'get_detailed_matrix_info'):
                    info = model.get_detailed_matrix_info()
                    
                    for i, layer_info in enumerate(info['layer_infos']):
                        if f'stable_rank_layer_{i}' not in layer_metrics:
                            layer_metrics[f'stable_rank_layer_{i}'] = []
                            layer_metrics[f'spectral_bound_layer_{i}'] = []
                            layer_metrics[f'spectral_bound_act_{i}'] = []
                            layer_metrics[f'rms_norm_layer_{i}'] = []
                        
                        # Convert tensors to CPU/numpy/scalar values for storage
                        stable_rank_val = layer_info['stable_rank']
                        if torch.is_tensor(stable_rank_val):
                            stable_rank_val = stable_rank_val.cpu().item()
                        
                        linear_spectral_val = layer_info['linear_spectral_norm']
                        if torch.is_tensor(linear_spectral_val):
                            linear_spectral_val = linear_spectral_val.cpu().item()
                            
                        activation_spectral_val = layer_info['activation_spectral_norm']
                        if torch.is_tensor(activation_spectral_val):
                            activation_spectral_val = activation_spectral_val.cpu().item()
                        
                        layer_metrics[f'stable_rank_layer_{i}'].append(stable_rank_val)
                        layer_metrics[f'spectral_bound_layer_{i}'].append(linear_spectral_val)
                        layer_metrics[f'spectral_bound_act_{i}'].append(activation_spectral_val)
                        
                        # Add to wandb log
                        log_dict[f'stable_rank_layer_{i}'] = stable_rank_val
                        log_dict[f'spectral_bound_layer_{i}'] = linear_spectral_val
                        log_dict[f'spectral_bound_act_{i}'] = activation_spectral_val

                        rms_norm_val = layer_info['rms_norm']
                        layer_metrics[f'rms_norm_layer_{i}'].append(rms_norm_val)
                        log_dict[f'rms_norm_layer_{i}'] = rms_norm_val
                                    
                    if 'end_to_end_bound' not in layer_metrics:
                        layer_metrics['end_to_end_bound'] = []
                    
                    # Convert end-to-end bound tensor to CPU/scalar
                    end_to_end_val = info['end_to_end_spectral_bound']
                    if torch.is_tensor(end_to_end_val):
                        end_to_end_val = end_to_end_val.cpu().item()
                    
                    layer_metrics['end_to_end_bound'].append(end_to_end_val)
                    log_dict['end_to_end_bound'] = end_to_end_val
                    
                    # --- NEW: Log distance to oracle ---
                    # Signed distance (bound - oracle)
                    diff_max_signed = end_to_end_val - oracle_lipschitz_max
                    diff_mean_signed = end_to_end_val - oracle_lipschitz_mean
                    
                    # Absolute distance |bound - oracle|
                    diff_max_abs = abs(diff_max_signed)
                    diff_mean_abs = abs(diff_mean_signed)
                    
                    log_dict.update({
                        'lipschitz_dist/signed_diff_vs_max_oracle': diff_max_signed,
                        'lipschitz_dist/abs_diff_vs_max_oracle': diff_max_abs,
                        'lipschitz_dist/signed_diff_vs_mean_oracle': diff_mean_signed,
                        'lipschitz_dist/abs_diff_vs_mean_oracle': diff_mean_abs
                    })

                    # --- NEW: Log distance to theoretical upper bound ---
                    # Signed distance (bound - theoretical max)
                    diff_theoretical_signed = end_to_end_val - theoretical_max_lipschitz
                    # Absolute distance |bound - theoretical max|
                    diff_theoretical_abs = abs(diff_theoretical_signed)
                    log_dict.update({
                        'lipschitz_dist/signed_diff_vs_theoretical_max': diff_theoretical_signed,
                        'lipschitz_dist/abs_diff_vs_theoretical_max': diff_theoretical_abs,
                    })

                    # Check if the model has Fourier features before trying to log them
                    if info.get('fourier_matrix_rms_norm') is not None:
                        fourier_rms_val = info['fourier_matrix_rms_norm']
                        oracle_rms_val = lipschitz_estimate['combined']['rms_gradient_mean']

                        # Signed distance (fourier_rms - oracle_rms)
                        diff_rms_signed = fourier_rms_val - oracle_rms_val

                        # Absolute distance |fourier_rms - oracle_rms|
                        diff_rms_abs = abs(diff_rms_signed)

                        log_dict.update({
                            'rms_norm/fourier_matrix_rms_norm': fourier_rms_val,
                            'rms_norm/oracle_rms_mean': oracle_rms_val,
                            'rms_norm_dist/signed_diff_vs_oracle_rms': diff_rms_signed,
                            'rms_norm_dist/abs_diff_vs_oracle_rms': diff_rms_abs
                        })

                
                wandb.log(log_dict)
    
    return model, metrics, layer_metrics, img, lpips_model

def plot_metrics(metrics, layer_metrics, args):
    """Create and save separate plots for different metrics, logging each to wandb"""
    
    # 1. PSNR plot
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['epochs'], metrics['psnr'], 'b-', linewidth=2)
    plt.title('PSNR Over Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"plots/psnr": wandb.Image(plt)})
    plt.close()
    
    # 2. SSIM plot
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['epochs'], metrics['ssim'], 'g-', linewidth=2)
    plt.title('SSIM Over Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"plots/ssim": wandb.Image(plt)})
    plt.close()
    
    # 3. LPIPS plot
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['epochs'], metrics['lpips'], 'r-', linewidth=2)
    plt.title('LPIPS Over Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('LPIPS (lower is better)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"plots/lpips": wandb.Image(plt)})
    plt.close()
    
    # 4. Layer Spectral Bounds
    if any('spectral_bound_layer_' in key for key in layer_metrics):
        plt.figure(figsize=(10, 6))
        for key in layer_metrics:
            if 'spectral_bound_layer_' in key:
                layer_num = key.split('_')[-1]
                plt.plot(metrics['epochs'], layer_metrics[key], label=f'Layer {layer_num}', linewidth=2)
        plt.title('Layer Spectral Bounds', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Spectral Norm', fontsize=12)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"plots/layer_spectral_bounds": wandb.Image(plt)})
        plt.close()
    
    # 5. Activation Spectral Bounds
    if any('spectral_bound_act_' in key for key in layer_metrics):
        plt.figure(figsize=(10, 6))
        for key in layer_metrics:
            if 'spectral_bound_act_' in key:
                layer_num = key.split('_')[-1]
                plt.plot(metrics['epochs'], layer_metrics[key], label=f'Activation {layer_num}', linewidth=2)
        plt.title('Activation Spectral Bounds', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Spectral Norm', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"plots/activation_spectral_bounds": wandb.Image(plt)})
        plt.close()
    
    # 6. Stable Ranks
    if any('stable_rank_layer_' in key for key in layer_metrics):
        plt.figure(figsize=(10, 6))
        for key in layer_metrics:
            if 'stable_rank_layer_' in key:
                layer_num = key.split('_')[-1]
                plt.plot(metrics['epochs'], layer_metrics[key], label=f'Layer {layer_num}', linewidth=2)
        plt.title('Stable Ranks', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Stable Rank', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"plots/stable_ranks": wandb.Image(plt)})
        plt.close()
    
    # 7. End-to-End Spectral Bound
    if 'end_to_end_bound' in layer_metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(metrics['epochs'], layer_metrics['end_to_end_bound'], 'r-', linewidth=2)
        plt.title('End-to-End Spectral Bound', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Spectral Bound', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"plots/end_to_end_bound": wandb.Image(plt)})
        plt.close()

    # 8. Layer RMS Norms
    # Check if the dictionary contains any key related to RMS norm
    if any('rms_norm_layer_' in key for key in layer_metrics):
        plt.figure(figsize=(10, 6))
        for key in layer_metrics:
            if 'rms_norm_layer_' in key:
                layer_num = key.split('_')[-1]
                plt.plot(metrics['epochs'], layer_metrics[key], label=f'Layer {layer_num}', linewidth=2)
        plt.title('Layer RMS Norms', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMS Norm', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"plots/layer_rms_norms": wandb.Image(plt)})
        plt.close()
    
    print(f"All plots saved and logged to wandb under 'plots/' namespace")

# at top of file
from pathlib import Path
import sys, os, io, atexit

def setup_console_logs(run_dir, run_id, combine=True):
    """
    Route Python prints to a per-run .out (and optionally .err) file,
    while still echoing to the terminal (tee behavior).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir/f"{run_id}.out"
    err_path = out_path if combine else run_dir/f"{run_id}.err"

    class Tee(io.TextIOBase):
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()
            return len(s)
        def flush(self):
            for st in self.streams: st.flush()

    # keep originals so you still see logs in the Slurm stream
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    f_out = open(out_path, "a", buffering=1)  # line-buffered
    f_err = f_out if combine else open(err_path, "a", buffering=1)
    sys.stdout = Tee(orig_stdout, f_out)
    sys.stderr = Tee(orig_stderr, f_err)

    atexit.register(lambda: [f.close() for f in {f_out, f_err} if f not in (orig_stdout, orig_stderr)])
    return out_path, err_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='000153.jpg', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--log_n_epochs', default=250, type=int)
    parser.add_argument('--norm_type', choices=['spectral_norm', 'bjoerck', 'sll'], default='bjoerck', type=str)
    parser.add_argument('--k_budget', default=0, type=float)
    parser.add_argument('--allocation', choices=['uniform', 'linear', 'exponential', 'cosine', 'midheavy', 'first'], default='exponential')
    parser.add_argument('--r_budget', default=2.0, type=float)
    parser.add_argument('--inpainting_ratio', default=0.125, type=float, 
                       help='Ratio of pixels to train on for inpainting (e.g., 0.125 for 12.5%)')
    parser.add_argument('--log_dir', default='../logs/inpainting/', type=str)
    parser.add_argument('--data_root', default='../data/celeba/', type=str)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--mapping_size', default=64, type=int)
    parser.add_argument('--model', choices=['relu_mlp', 'relu_ffn', 'gauss_ffn', 'gauss_mlp', 'siren_mlp', 'groupsort_ffn', 'relu_posencoding', 'groupsort_mlp'], default='relu_ffn')
    parser.add_argument('--log_image_evolution', action='store_true', 
                       help='Log image evolution during training (can be memory intensive)')
    
    parser.add_argument('--seed', default=12, type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    if 'mlp' in args.model:
        args.num_layers = args.num_layers + 1
    set_seed(args.seed)

    # Build a deterministic, filesystem-safe id
    stem = Path(args.image).stem
    run_id = f"model-{args.model}_k-{args.k_budget}_r-{args.r_budget}_a-{args.allocation}_img-{stem}_norm-{args.norm_type}_layers-{args.num_layers}_hidden-{args.hidden_dim}"

    # Setup console logging
    out_path, err_path = setup_console_logs(run_dir = args.log_dir, run_id=run_id, combine=True)
    wandb.init(
        project="image_overfitting_constrained_siren",
        dir = args.log_dir,
        id=run_id,                 # controls the folder suffix
        name=run_id,               # display name in UI
        config=vars(args)
    )

    model, metrics, layer_metrics, original_img, lpips_model = train_model(args)

    print(model)
    
    # Final reconstruction
    with torch.no_grad():
        h, w, c = original_img.shape
        coords = get_coordinates(h, w).to(next(model.parameters()).device)
        final_pred = model(coords).view(h, w, c).cpu()
        
        final_psnr = psnr(final_pred, original_img).item()
        final_ssim = ssim(original_img.numpy(), final_pred.numpy(), channel_axis=-1, data_range=1.0)
        
        # Compute final LPIPS
        final_lpips = compute_lpips(original_img.numpy(), final_pred.numpy(), 
                                  lpips_model, next(model.parameters()).device)
        
        print(f"Final PSNR: {final_psnr:.2f}")
        print(f"Final SSIM: {final_ssim:.4f}")
        print(f"Final LPIPS: {final_lpips:.4f}")
        
        # Create final comparison image
        final_comparison_fig = create_comparison_image(
            original_img.numpy(), 
            final_pred.numpy(), 
            args.epochs, 
            final_psnr, 
            final_ssim,
            final_lpips
        )
        
        # Log final metrics and images to wandb
        log_dict = {
            'image_metrics/final_psnr': final_psnr,
            'image_metrics/final_ssim': final_ssim,
            'image_metrics/final_lpips': final_lpips,
            'images/final_comparison': wandb.Image(final_comparison_fig, 
                                                 caption=f'Final Result: PSNR={final_psnr:.2f}, SSIM={final_ssim:.4f}, LPIPS={final_lpips:.4f}'),
            'images/final_reconstruction': wandb.Image(np.clip(final_pred.numpy(), 0, 1), caption='Final Reconstruction')
        }
            
        wandb.log(log_dict)
        plt.close(final_comparison_fig)
    
    plot_metrics(metrics, layer_metrics, args)
    
    wandb.finish()

if __name__ == "__main__":
    main()