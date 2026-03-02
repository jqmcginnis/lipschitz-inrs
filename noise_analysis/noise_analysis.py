import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

sns.set_theme(style='darkgrid', context="notebook", font_scale=1.0)
sns.set_palette('viridis')

def estimate_lipschitz_rectangular(image_array, x_range=(-1, 1), y_range=(-1, 1)):
    I = image_array.astype(np.float64)
    if I.max() > 1.0: I = I / 255.0
    
    H, W = I.shape
    grad_y_pixel, grad_x_pixel = np.gradient(I, edge_order=2)
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    dx_coord = (x_max - x_min) / W
    dy_coord = (y_max - y_min) / H
    
    grad_x_scaled = grad_x_pixel / dx_coord
    grad_y_scaled = grad_y_pixel / dy_coord
    
    gradient_magnitude = np.sqrt(grad_x_scaled**2 + grad_y_scaled**2)
    lipschitz_constant = gradient_magnitude.max()
    
    return lipschitz_constant, grad_x_scaled, grad_y_scaled, gradient_magnitude

def estimate_rgb_lipschitz_rectangular(image_array, x_range=(-1, 1), y_range=(-1, 1)):
    if image_array.ndim != 3 or image_array.shape[2] not in (3, 4):
        if image_array.ndim == 2:
             L, gx, gy, gm = estimate_lipschitz_rectangular(image_array)
             return {'stats': {'max': L, 'p99_9': np.percentile(gm, 99.9), 'p99_99': np.percentile(gm, 99.99)}, 'all_grads': gm.flatten()}
        raise ValueError("Expected RGB image")
        
    if image_array.shape[2] == 4:
        image_array = image_array[..., :3]

    all_magnitudes = []
    
    for i in range(3):
        _, _, _, gm = estimate_lipschitz_rectangular(image_array[..., i], x_range, y_range)
        all_magnitudes.append(gm.flatten())

    global_grads = np.concatenate(all_magnitudes)
    
    gxR, gxG, gxB = [estimate_lipschitz_rectangular(image_array[..., i])[1] for i in range(3)]
    gyR, gyG, gyB = [estimate_lipschitz_rectangular(image_array[..., i])[2] for i in range(3)]
    
    a = gxR**2 + gxG**2 + gxB**2
    b = gxR*gyR + gxG*gyG + gxB*gyB
    d = gyR**2 + gyG**2 + gyB**2
    sigma_max = np.sqrt(0.5 * ((a+d) + np.sqrt((a-d)**2 + 4*b**2)))
    lipschitz_spectral = float(np.max(sigma_max))

    stats = {
        'max': lipschitz_spectral,
        'p99_9': np.percentile(global_grads, 99.9),
        'p99_99': np.percentile(global_grads, 99.99)
    }

    return {'stats': stats, 'all_grads': global_grads}

def add_poisson_readout_noise(image_array, max_photons=30, readout_noise=2):
    img = image_array.astype(np.float64)
    if img.max() > 1.0: img /= 255.0
    
    img_photons = img * max_photons
    noisy_photons = np.random.poisson(img_photons).astype(np.float64)
    noisy_signal = noisy_photons + np.random.normal(0, readout_noise, noisy_photons.shape)
    
    noisy_norm = np.clip(noisy_signal, 0, None) / max_photons
    return np.clip(noisy_norm, 0.0, 1.0)

def create_6x4_grid_summary(image_dir, target_files, save_path):
    noise_levels = [30, 90, 120] 
    readout_noise = 2
    
    fig, axes = plt.subplots(6, 4, figsize=(12, 15), constrained_layout=True)
    
    palette = sns.color_palette("viridis", n_colors=5)
    color_clean = palette[0]
    color_noisy = palette[-2]
    
    print(f"Generating 6x4 Summary (Global Sync per Image)...")

    row_idx = 0

    for filename in target_files:
        file_path = os.path.join(image_dir, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {filename} (not found).")
            row_idx += 3
            continue

        img_pil = Image.open(file_path).convert('RGB')
        # Changed to asarray to avoid copy warning
        img_clean = np.asarray(img_pil) 
        res_clean = estimate_rgb_lipschitz_rectangular(img_clean)
        
        current_image_hist_axes = []

        for photons in noise_levels:
            img_noisy = add_poisson_readout_noise(img_clean, max_photons=photons, readout_noise=readout_noise)
            res_noisy = estimate_rgb_lipschitz_rectangular(img_noisy)
            
            # Images
            axes[row_idx, 0].imshow(img_clean)
            axes[row_idx, 0].set_title(f"Original ", fontsize=10, fontweight='bold')
            axes[row_idx, 0].axis('off')
            
            axes[row_idx, 1].imshow(img_noisy)
            axes[row_idx, 1].set_title(f"Noisy ($\gamma$={photons})", fontsize=10, fontweight='bold')
            axes[row_idx, 1].axis('off')
            
            # Histograms
            configs = [
                (res_clean, color_clean, "Original Gradients", axes[row_idx, 2]),
                (res_noisy, color_noisy, f"Noisy Gradients ($\gamma$={photons})", axes[row_idx, 3])
            ]
            
            for res, color, label, ax in configs:
                grads = res['all_grads']
                stats = res['stats']
                
                sns.histplot(grads, ax=ax, kde=True, color=color, 
                             stat='density', bins=50, alpha=0.7, linewidth=0)
                
                lines_info = [
                    (stats['p99_9'], '#e74c3c', '99.9%'),
                    (stats['p99_99'], '#e67e22', '99.99%'),
                    (stats['max'],    '#8e44ad', 'Max')
                ]

                for val, line_color, line_label in lines_info:
                    legend_label = f"{val:.1f}"
                    ax.axvline(val, color=line_color, linestyle='--', linewidth=1.5, label=legend_label)
                
                ax.set_title(label, fontsize=10, fontweight='bold')
                ax.set_yscale('log')
                
                if row_idx == 5: 
                    ax.set_xlabel("||∇I||", fontsize=9)
                else:
                    ax.set_xlabel("")
                    
                ax.legend(fontsize=7, loc='upper right')
                current_image_hist_axes.append(ax)

            row_idx += 1

        # Sync axes per image
        y_mins = [ax.get_ylim()[0] for ax in current_image_hist_axes]
        y_maxs = [ax.get_ylim()[1] for ax in current_image_hist_axes]
        global_ymin = min(y_mins)
        global_ymax = max(y_maxs)

        x_maxs = [ax.get_xlim()[1] for ax in current_image_hist_axes]
        global_xmax = max(x_maxs)

        for ax in current_image_hist_axes:
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_xlim(0, global_xmax)

    plt.savefig(save_path, dpi=100) 
    plt.close(fig)
    print(f"Saved 6x4 grid to: {save_path}")

if __name__ == "__main__":
    
    example_folder = "."
    
    target_files = [
        "celeba_ex_0001.png",   
        "celeba_ex_0002.png"  
    ]
    
    if os.path.exists(example_folder):
        output_file = os.path.join(example_folder, "celeba_6x4_shared_y.png")
        
        create_6x4_grid_summary(
            image_dir=example_folder,
            target_files=target_files,
            save_path=output_file
        )
    else:
        print(f"Error: Folder not found at {example_folder}")
