import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import PIL

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import PIL.Image

def estimate_lipschitz_rectangular(image_array, x_range=(-1, 1), y_range=(-1, 1)):
    """
    Estimate Lipschitz constant for images on rectangular domains.
    
    Parameters:
    -----------
    image_array : numpy.ndarray
        2D array representing the image
    x_range : tuple
        (x_min, x_max) coordinate range for horizontal axis
    y_range : tuple  
        (y_min, y_max) coordinate range for vertical axis
    
    Returns:
    --------
    lipschitz_constant : float
        Estimated Lipschitz constant
    grad_x_scaled : numpy.ndarray
        Scaled x-gradients
    grad_y_scaled : numpy.ndarray  
        Scaled y-gradients
    gradient_magnitude : numpy.ndarray
        Combined gradient magnitudes
    """
    # Normalize image to [0,1]
    I = image_array.astype(np.float64)
    if I.max() > 1.0:
        I = I / 255.0
    
    H, W = I.shape
    
    # Compute pixel-space gradients
    grad_y_pixel, grad_x_pixel = np.gradient(I, edge_order=2)  # dI/dpixel
    
    # Compute coordinate scaling factors for each direction
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Scale factors: convert from pixel differences to coordinate differences
    dx_coord = (x_max - x_min) / W  # coordinate distance per pixel in x
    dy_coord = (y_max - y_min) / H  # coordinate distance per pixel in y
    
    # Scale gradients to coordinate space
    # grad_pixel = dI/dpixel, grad_coord = dI/dcoord = (dI/dpixel) * (dpixel/dcoord)
    grad_x_scaled = grad_x_pixel / dx_coord
    grad_y_scaled = grad_y_pixel / dy_coord
    
    # Compute gradient magnitude (Euclidean norm)
    gradient_magnitude = np.sqrt(grad_x_scaled**2 + grad_y_scaled**2)
    
    # Lipschitz constant is the maximum gradient magnitude
    lipschitz_constant = gradient_magnitude.max()
    
    #print(f"Coordinate step sizes: Δx = {dx_coord:.6f}, Δy = {dy_coord:.6f}")
    #print(f"Estimated Lipschitz constant: {lipschitz_constant:.6f}")
    #print(f"Mean gradient magnitude: {gradient_magnitude.mean():.6f}")
    
    return lipschitz_constant, grad_x_scaled, grad_y_scaled, gradient_magnitude

import numpy as np

def estimate_rgb_lipschitz_rectangular(image_array, x_range=(-1, 1), y_range=(-1, 1)):
    if image_array.ndim != 3 or image_array.shape[2] not in (3, 4):
        raise ValueError("Expected RGB(A) image with shape (H, W, 3|4)")
    if image_array.shape[2] == 4:
        # drop alpha
        image_array = image_array[..., :3]

    channel_names = ['Red', 'Green', 'Blue']
    per_channel = {}

    # Per-channel grads
    gxs, gys = [], []
    lips = []
    means = []
    rms_list = []

    for i, name in enumerate(channel_names):
        L, gx, gy, gm = estimate_lipschitz_rectangular(
            image_array[..., i], x_range, y_range
        )
        per_channel[name] = {
            'lipschitz_constant': L,
            'grad_x': gx,
            'grad_y': gy,
            'gradient_magnitude': gm,
            'mean_gradient': float(gm.mean()),
            'std_gradient': float(gm.std()),
            'rms_gradient': float(np.sqrt(np.mean(gm**2))),
        }
        gxs.append(gx); gys.append(gy)
        lips.append(L); means.append(gm.mean()); rms_list.append(np.sqrt(np.mean(gm**2)))

    # --- Correct combined Lipschitz (spectral norm of J) ---
    gxR, gxG, gxB = gxs
    gyR, gyG, gyB = gys
    a = gxR*gxR + gxG*gxG + gxB*gxB               # sum_c (∂c/∂x)^2
    b = gxR*gyR + gxG*gyG + gxB*gyB               # sum_c (∂c/∂x)(∂c/∂y)
    d = gyR*gyR + gyG*gyG + gyB*gyB               # sum_c (∂c/∂y)^2
    trace = a + d
    delta = np.sqrt((a - d)**2 + 4.0*b*b)
    sigma_max = np.sqrt(0.5*(trace + delta))      # largest singular value field
    lipschitz_spectral = float(np.max(sigma_max))  # <-- use this as RGB Lipschitz (Euclidean)

    combined = {
        'lipschitz_spectral': lipschitz_spectral,      # correct combined (Euclidean)
        'lipschitz_max': float(np.max(lips)),  # your previous notion (ℓ∞ in color)
        'lipschitz_mean': float(np.mean(lips)),
        'lipschitz_std': float(np.std(lips)),
        'mean_gradient_max': float(np.max(means)),
        'mean_gradient_mean': float(np.mean(means)),
        'rms_gradient_mean': float(np.mean(rms_list)),
    }

    return {'per_channel': per_channel, 'combined': combined,
            'domain': {'x_range': x_range, 'y_range': y_range}}


def visualize_lipschitz_analysis(image_array, lipschitz_const, grad_x, grad_y, grad_mag, 
                                x_range=(-1, 1), y_range=(0, 1)):
    """Create comprehensive visualization of Lipschitz analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    im0 = axes[0,0].imshow(image_array, cmap='gray', extent=[*x_range, *y_range], 
                           origin='lower', aspect='auto')
    axes[0,0].set_title('Original Image')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0,0])
    
    # X-gradient
    im1 = axes[0,1].imshow(grad_x, cmap='RdBu_r', extent=[*x_range, *y_range], 
                           origin='lower', aspect='auto')
    axes[0,1].set_title('∂I/∂x (scaled)')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,1])
    
    # Y-gradient  
    im2 = axes[0,2].imshow(grad_y, cmap='RdBu_r', extent=[*x_range, *y_range], 
                           origin='lower', aspect='auto')
    axes[0,2].set_title('∂I/∂y (scaled)')
    axes[0,2].set_xlabel('x')
    axes[0,2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,2])
    
    # Gradient magnitude
    im3 = axes[1,0].imshow(grad_mag, cmap='hot', extent=[*x_range, *y_range], 
                           origin='lower', aspect='auto')
    axes[1,0].set_title(f'‖∇I‖ (L = {lipschitz_const:.4f})')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Gradient histogram
    axes[1,1].hist(grad_mag.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(lipschitz_const, color='red', linestyle='--', linewidth=2,
                     label=f'Max = {lipschitz_const:.4f}')
    axes[1,1].axvline(grad_mag.mean(), color='blue', linestyle='--', linewidth=2,
                     label=f'Mean = {grad_mag.mean():.4f}')
    axes[1,1].set_xlabel('Gradient Magnitude')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Gradient Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_yscale('log')
    
    # Cumulative distribution
    sorted_grads = np.sort(grad_mag.flatten())
    cdf = np.arange(1, len(sorted_grads) + 1) / len(sorted_grads)
    axes[1,2].plot(sorted_grads, cdf)
    axes[1,2].axvline(lipschitz_const, color='red', linestyle='--', 
                     label=f'Lipschitz = {lipschitz_const:.4f}')
    axes[1,2].set_xlabel('Gradient Magnitude')
    axes[1,2].set_ylabel('Cumulative Probability')
    axes[1,2].set_title('Gradient CDF')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()

def visualize_rgb_lipschitz(image_array, rgb_results, x_range=(-1, 1), y_range=(-1, 1)):
    """Create visualization for RGB Lipschitz analysis and return figures as images."""

    # --- First figure ---
    fig1, axes = plt.subplots(3, 4, figsize=(20, 15))
    channel_names = ['Red', 'Green', 'Blue']
    colors = ['Reds', 'Greens', 'Blues']

    for i, (channel_name, cmap) in enumerate(zip(channel_names, colors)):
        results = rgb_results['per_channel'][channel_name]

        # Original channel
        im0 = axes[i,0].imshow(image_array[:,:,i], cmap='gray',
                              extent=[*x_range, *y_range], origin='lower', aspect='auto')
        axes[i,0].set_title(f'{channel_name} Channel')
        axes[i,0].set_ylabel('y')
        if i == 2: axes[i,0].set_xlabel('x')
        plt.colorbar(im0, ax=axes[i,0])

        # X gradient
        im1 = axes[i,1].imshow(results['grad_x'], cmap='RdBu_r',
                              extent=[*x_range, *y_range], origin='lower', aspect='auto')
        axes[i,1].set_title(f'∂{channel_name}/∂x')
        if i == 2: axes[i,1].set_xlabel('x')
        plt.colorbar(im1, ax=axes[i,1])

        # Y gradient
        im2 = axes[i,2].imshow(results['grad_y'], cmap='RdBu_r',
                              extent=[*x_range, *y_range], origin='lower', aspect='auto')
        axes[i,2].set_title(f'∂{channel_name}/∂y')
        if i == 2: axes[i,2].set_xlabel('x')
        plt.colorbar(im2, ax=axes[i,2])

        # Gradient magnitude
        im3 = axes[i,3].imshow(results['gradient_magnitude'], cmap='hot',
                              extent=[*x_range, *y_range], origin='lower', aspect='auto')
        axes[i,3].set_title(f'‖∇{channel_name}‖ (L={results["lipschitz_constant"]:.4f})')
        if i == 2: axes[i,3].set_xlabel('x')
        plt.colorbar(im3, ax=axes[i,3])

    plt.tight_layout()

    # --- Second figure ---
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Lipschitz constants by channel
    lipschitz_values = [rgb_results['per_channel'][ch]['lipschitz_constant']
                       for ch in channel_names]
    bars = axes[0].bar(channel_names, lipschitz_values, color=['red', 'green', 'blue'], alpha=0.7)
    axes[0].set_ylabel('Lipschitz Constant')
    axes[0].set_title('Lipschitz Constants by Channel')
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, lipschitz_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lipschitz_values)*0.01,
                    f'{val:.4f}', ha='center', va='bottom')

    # Combined gradient histogram
    for i, (channel_name, color) in enumerate(zip(channel_names, ['red', 'green', 'blue'])):
        grad_mag = rgb_results['per_channel'][channel_name]['gradient_magnitude']
        axes[1].hist(grad_mag.flatten(), bins=50, alpha=0.5, label=channel_name,
                    color=color, edgecolor='black', linewidth=0.5)

    axes[1].axvline(rgb_results['combined']['lipschitz_max'], color='black',
                   linestyle='--', linewidth=2, label=f'Max L = {rgb_results["combined"]["lipschitz_max"]:.4f}')
    axes[1].set_xlabel('Gradient Magnitude')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Gradient Distributions by Channel')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # Channel statistics
    stats = ['lipschitz_constant', 'mean_gradient', 'std_gradient']
    stat_labels = ['Lipschitz L', 'Mean |∇I|', 'Std |∇I|']

    x_pos = np.arange(len(channel_names))
    width = 0.25

    for i, (stat, label) in enumerate(zip(stats, stat_labels)):
        values = [rgb_results['per_channel'][ch][stat] for ch in channel_names]
        axes[2].bar(x_pos + i*width, values, width, label=label, alpha=0.8)

    axes[2].set_xlabel('Channel')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Channel Statistics Comparison')
    axes[2].set_xticks(x_pos + width)
    axes[2].set_xticklabels(channel_names)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # --- Convert figures to images ---
    def fig_to_image(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        return PIL.Image.open(buf)

    image1 = fig_to_image(fig1)
    image2 = fig_to_image(fig2)

    plt.close(fig1)
    plt.close(fig2)

    return image1, image2


def ask_oracle(image_path, x_range=(-1, 1), y_range=(-1, 1), try_rgb=True):
    """
    Complete analysis of an image file - handles both grayscale and RGB.
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    x_range, y_range : tuple
        Coordinate ranges for the domain
    try_rgb : bool
        Whether to attempt RGB analysis
    """
    try:
        # Try to load as RGB first
        img = Image.open(image_path)
        img_array = np.array(img)
        
        print(f"Loaded image: {image_path}")
        print(f"Image mode: {img.mode}")
        print(f"Array shape: {img_array.shape}")
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3 and try_rgb:
            print("\n" + "="*60)
            print("RGB ANALYSIS")
            print("="*60)
            
            # RGB analysis
            rgb_results = estimate_rgb_lipschitz_rectangular(img_array, x_range, y_range)
            print(f"Max Lipschitz constant across channels: {rgb_results['combined']['lipschitz_max']:.6f}")
            print(f"Spectral Lipschitz constant (Euclidean): {rgb_results['combined']['lipschitz_spectral']:.6f}")
            # visualize_rgb_lipschitz(img_array, rgb_results, x_range, y_range)
            
            # Also do grayscale version for comparison
            print("\n" + "="*60)
            print("GRAYSCALE ANALYSIS (for comparison)")
            print("="*60)
            gray_img = img.convert('L')
            gray_array = np.array(gray_img)
            
        else:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray_img = img.convert('L')
                gray_array = np.array(gray_img)
            else:
                gray_array = img_array
        
        # Grayscale analysis
        L, grad_x, grad_y, grad_mag = estimate_lipschitz_rectangular(gray_array, x_range, y_range)
        print(f"Estimated Lipschitz constant (grayscale): {L:.6f}")
        # visualize_lipschitz_analysis(gray_array, L, grad_x, grad_y, grad_mag, x_range, y_range)
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Could not find image file '{image_path}'")
        return False
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False