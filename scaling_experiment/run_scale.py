import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import argparse
import wandb
import csv

from models_unconstrained import SirenMLP

#  --- Helper Functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)

def get_coordinates(h, w, inpainting_ratio=0.125):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    
    num_pixels = h * w
    num_train_pixels = int(num_pixels * inpainting_ratio)
    
    train_indices = torch.randperm(num_pixels)[:num_train_pixels]
    test_mask = ~torch.isin(torch.arange(num_pixels), train_indices)
    test_indices = torch.arange(num_pixels)[test_mask]
    
    return coords, train_indices, test_indices

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def parse_arguments():
    parser = argparse.ArgumentParser(description='SirenMLP Image Inpainting')
    
    # Data arguments
    parser.add_argument('--image', type=str, default='000153.jpg',
                       help='Path to input image')
    parser.add_argument('--inpainting_ratio', type=float, default=0.25,
                       help='Ratio of pixels to use for training (default: 0.25)')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension of the MLP (default: 256)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of hidden layers (default: 4)')
    parser.add_argument('--alpha', type=float, default=1.5,
                       help='Alpha parameter for Siren (default: 30)')
    parser.add_argument('--omega', type=float, default=30,
                       help='Omega parameter for Siren (default: 30)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of training epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--log_n_epochs', type=int, default=100,
                       help='Log metrics every N epochs (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default='siren-inpainting',
                       help='Weights & Biases project name (default: siren-inpainting)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases run name (default: auto-generated)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity/username')
    
    # Output arguments
    parser.add_argument('--data_root', default='../data/celeba/', type=str)
    parser.add_argument('--output_dir', type=str, default='../logs/scaling',
                       help='Output directory for saved files (default: current directory)')
    
    return parser.parse_args()

# --- Main Training and Plotting Loop ---
def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Initialize wandb if enabled
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img = load_image(args.data_root+args.image)
    h, w, c = img.shape
    print(f"Image shape: {h}x{w}x{c}")
    
    all_coords, train_indices, test_indices = get_coordinates(h, w, args.inpainting_ratio)
    
    # Prepare data for training and testing
    train_coords = all_coords[train_indices].to(device)
    test_coords = all_coords[test_indices].to(device)
    
    train_target = img.view(-1, c)[train_indices].to(device)
    test_target = img.view(-1, c)[test_indices].to(device)
    full_target = img.view(-1, c).to(device)  # For full PSNR calculation
    
    print(f"Training pixels: {len(train_indices)} ({args.inpainting_ratio:.1%})")
    print(f"Test pixels: {len(test_indices)} ({1-args.inpainting_ratio:.1%})")

    # Initialize model
    model = SirenMLP(
        input_dim=2, 
        hidden_dim=args.hidden_dim, 
        output_dim=c, 
        num_layers=args.num_layers, 
        alpha=args.alpha,
        omega=args.omega
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Log model info to wandb
    if use_wandb:
        wandb.watch(model, log='all', log_freq=args.log_n_epochs)
    
    # Lists to store metrics for plotting
    train_psnrs = []
    test_psnrs = []
    full_psnrs = []  # Added for full PSNR
    spectral_bounds = []
    log_epochs = []
    
    # Prepare CSV file for logging
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    csv_filename = os.path.join(args.output_dir, f"{image_name}_{args.alpha}_{args.inpainting_ratio}_{args.omega}.csv")
    
    # Initialize CSV with headers
    csv_headers = ['epoch', 'train_psnr', 'test_psnr', 'full_psnr', 'spectral_bound']
    
    # Add layer-specific headers
    model.eval()
    with torch.no_grad():
        temp_detailed_info = model.get_detailed_matrix_info()
        num_layers = len(temp_detailed_info['layer_infos'])
        
        for i in range(num_layers):
            csv_headers.extend([
                f'layer_{i}_linear_spectral_norm',
                f'layer_{i}_activation_spectral_norm', 
                f'layer_{i}_combined_spectral_norm',
                f'layer_{i}_frobenius_norm',
                f'layer_{i}_stable_rank',
                f'layer_{i}_spectral_condition_no'
            ])
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_coords)
        loss = F.mse_loss(pred, train_target)
        loss.backward()
        optimizer.step()
        
        # Log every epoch to wandb (if enabled)
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': loss.item(),
            }, step=epoch + 1)
        
        if (epoch + 1) % args.log_n_epochs == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on training, test, and full sets
                train_pred = model(train_coords)
                test_pred = model(test_coords)
                full_pred = model(all_coords.to(device))  # Full prediction for all pixels
                
                train_psnr_val = psnr(train_pred, train_target).item()
                test_psnr_val = psnr(test_pred, test_target).item()
                full_psnr_val = psnr(full_pred, full_target).item()  # Full PSNR
                
                # Get the end-to-end spectral bound
                spectral_bound_val = model.get_end_to_end_spectral_bound().item()
                
                # Get detailed matrix information for each layer
                detailed_info = model.get_detailed_matrix_info()
                layer_infos = detailed_info['layer_infos']
                
                train_psnrs.append(train_psnr_val)
                test_psnrs.append(test_psnr_val)
                full_psnrs.append(full_psnr_val)  # Store full PSNR
                spectral_bounds.append(spectral_bound_val)
                log_epochs.append(epoch + 1)
                
                print(f"Epoch {epoch+1:4d}: Loss={loss.item():.6f}, Train PSNR={train_psnr_val:.2f}dB, Test PSNR={test_psnr_val:.2f}dB, Full PSNR={full_psnr_val:.2f}dB, Spectral Bound={spectral_bound_val:.2f}")
                
                # Prepare CSV row data
                csv_row = [epoch + 1, train_psnr_val, test_psnr_val, full_psnr_val, spectral_bound_val]
                
                # Add layer-specific data to CSV row
                for layer_info in layer_infos:
                    csv_row.extend([
                        layer_info['linear_spectral_norm'],
                        layer_info['activation_spectral_norm'],
                        layer_info['combined_spectral_norm'],
                        layer_info['frobenius_norm'],
                        layer_info['stable_rank'],
                        layer_info['spectral_condition_no']
                    ])
                
                # Write to CSV (append mode, create headers if first write)
                file_exists = os.path.exists(csv_filename)
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(csv_headers)
                    writer.writerow(csv_row)
                
                # Log detailed metrics to wandb
                if use_wandb:
                    # Create log dictionary with basic metrics
                    log_dict = {
                        'train_psnr': train_psnr_val,
                        'test_psnr': test_psnr_val,
                        'full_psnr': full_psnr_val,  # Added full PSNR to wandb
                        'spectral_bound': spectral_bound_val,
                    }
                    
                    # Add layer-specific spectral norm information
                    for i, layer_info in enumerate(layer_infos):
                        layer_prefix = f'layer_{i}'
                        log_dict.update({
                            f'{layer_prefix}/linear_spectral_norm': layer_info['linear_spectral_norm'],
                            f'{layer_prefix}/activation_spectral_norm': layer_info['activation_spectral_norm'],
                            f'{layer_prefix}/combined_spectral_norm': layer_info['combined_spectral_norm'],
                            f'{layer_prefix}/frobenius_norm': layer_info['frobenius_norm'],
                            f'{layer_prefix}/stable_rank': layer_info['stable_rank'],
                            f'{layer_prefix}/spectral_condition_no': layer_info['spectral_condition_no'],
                        })
                    
                    wandb.log(log_dict, step=epoch + 1)

    # Plot the results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    
    # Plot 1: PSNR evolution (updated to include full PSNR)
    ax1.plot(log_epochs, train_psnrs, label='Train PSNR', color='blue', linewidth=2)
    ax1.plot(log_epochs, test_psnrs, label='Test PSNR', color='red', linewidth=2)
    ax1.plot(log_epochs, full_psnrs, label='Full PSNR', color='green', linewidth=2)  # Added full PSNR plot
    ax1.set_title('Train, Test, and Full PSNR Evolution', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot 2: Spectral Bound evolution
    ax2.plot(log_epochs, spectral_bounds, label='End-to-End Spectral Bound', color='purple', linewidth=2)
    ax2.set_title('End-to-End Spectral Bound Evolution', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Spectral Bound', fontsize=12)
    ax2.set_yscale('log') # Log scale is useful for spectral bounds
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    # Plot 3: Layer-wise Linear Spectral Norms (final values)
    model.eval()
    with torch.no_grad():
        final_detailed_info = model.get_detailed_matrix_info()
        final_layer_infos = final_detailed_info['layer_infos']
        
        layer_indices = list(range(len(final_layer_infos)))
        linear_spectral_norms = [info['linear_spectral_norm'] for info in final_layer_infos]
        combined_spectral_norms = [info['combined_spectral_norm'] for info in final_layer_infos]
        
        x_pos = np.arange(len(layer_indices))
        width = 0.35
        
        ax3.bar(x_pos - width/2, linear_spectral_norms, width, label='Linear Spectral Norm', alpha=0.8)
        ax3.bar(x_pos + width/2, combined_spectral_norms, width, label='Combined (Linear × Activation)', alpha=0.8)
        ax3.set_title('Final Layer-wise Spectral Norms', fontsize=16)
        ax3.set_xlabel('Layer Index', fontsize=12)
        ax3.set_ylabel('Spectral Norm', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Layer {i}' for i in layer_indices])
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)
        
    # Plot 4: Stable Rank and Condition Number
    stable_ranks = [info['stable_rank'] for info in final_layer_infos]
    condition_numbers = [info['spectral_condition_no'] for info in final_layer_infos]
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(layer_indices, stable_ranks, 'o-', color='green', linewidth=2, markersize=8, label='Stable Rank')
    line2 = ax4_twin.plot(layer_indices, condition_numbers, 's-', color='orange', linewidth=2, markersize=8, label='Condition Number')
    
    ax4.set_title('Layer-wise Stable Rank and Condition Number', fontsize=16)
    ax4.set_xlabel('Layer Index', fontsize=12)
    ax4.set_ylabel('Stable Rank', fontsize=12, color='green')
    ax4_twin.set_ylabel('Condition Number', fontsize=12, color='orange')
    ax4.set_xticks(layer_indices)
    ax4.set_xticklabels([f'Layer {i}' for i in layer_indices])
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.suptitle("SirenMLP: Performance, Stability, and Layer Analysis", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    plot_path = os.path.join(args.output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Log plot to wandb
    if use_wandb:
        wandb.log({'training_metrics_plot': wandb.Image(plot_path)})
    
    plt.close()
    
    # Generate and save inference image
    model.eval()
    with torch.no_grad():
        full_pred = model(all_coords.to(device))
        reconstructed_img = full_pred.view(h, w, c).cpu().numpy()
        reconstructed_img = np.clip(reconstructed_img, 0, 1)
        
        # Save the reconstructed image
        inferred_path = os.path.join(args.output_dir, 'inferred_image.png')
        reconstructed_pil = Image.fromarray((reconstructed_img * 255).astype(np.uint8))
        reconstructed_pil.save(inferred_path)
        
        # Create comparison image (original vs reconstructed)
        original_img = img.cpu().numpy()
        comparison = np.concatenate([original_img, reconstructed_img], axis=1)
        comparison_pil = Image.fromarray((comparison * 255).astype(np.uint8))
        comparison_path = os.path.join(args.output_dir, 'comparison.png')
        comparison_pil.save(comparison_path)
        
        # Log images to wandb
        if use_wandb:
            wandb.log({
                'original_image': wandb.Image(original_img, caption='Original'),
                'reconstructed_image': wandb.Image(reconstructed_img, caption='Reconstructed'),
                'comparison': wandb.Image(comparison, caption='Original vs Reconstructed')
            })
        
        # Final metrics
        final_train_psnr = psnr(model(train_coords), train_target).item()
        final_test_psnr = psnr(model(test_coords), test_target).item()
        final_full_psnr = psnr(full_pred, full_target).item()  # Final full PSNR
        final_spectral_bound = model.get_end_to_end_spectral_bound().item()
        
        # Get final detailed layer information
        final_detailed_info = model.get_detailed_matrix_info()
        final_layer_infos = final_detailed_info['layer_infos']
        
        print(f"\nFinal Results:")
        print(f"Final Train PSNR: {final_train_psnr:.2f}dB")
        print(f"Final Test PSNR: {final_test_psnr:.2f}dB")
        print(f"Final Full PSNR: {final_full_psnr:.2f}dB")  # Print final full PSNR
        print(f"Final Spectral Bound: {final_spectral_bound:.2f}")
        
        # Print layer-wise information
        print(f"\nLayer-wise Analysis:")
        for i, layer_info in enumerate(final_layer_infos):
            print(f"Layer {i}: Linear SN={layer_info['linear_spectral_norm']:.3f}, "
                  f"Combined SN={layer_info['combined_spectral_norm']:.3f}, "
                  f"Stable Rank={layer_info['stable_rank']:.2f}")
        
        if use_wandb:
            # Log final metrics
            final_log_dict = {
                'final_train_psnr': final_train_psnr,
                'final_test_psnr': final_test_psnr,
                'final_full_psnr': final_full_psnr,  # Added final full PSNR to wandb
                'final_spectral_bound': final_spectral_bound,
            }
            
            # Add final layer-specific information
            for i, layer_info in enumerate(final_layer_infos):
                layer_prefix = f'final_layer_{i}'
                final_log_dict.update({
                    f'{layer_prefix}/linear_spectral_norm': layer_info['linear_spectral_norm'],
                    f'{layer_prefix}/activation_spectral_norm': layer_info['activation_spectral_norm'],
                    f'{layer_prefix}/combined_spectral_norm': layer_info['combined_spectral_norm'],
                    f'{layer_prefix}/frobenius_norm': layer_info['frobenius_norm'],
                    f'{layer_prefix}/stable_rank': layer_info['stable_rank'],
                    f'{layer_prefix}/spectral_condition_no': layer_info['spectral_condition_no'],
                })
            
            wandb.log(final_log_dict)
        
        print(f"\nFiles saved to '{args.output_dir}':")
        print(f"- Training metrics plot: 'training_metrics.png'")
        print(f"- Inferred image: 'inferred_image.png'")
        print(f"- Comparison image: 'comparison.png'")
        print(f"- CSV log: '{csv_filename}'")  # Added CSV file info

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()