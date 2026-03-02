import numpy as np
import torch
import os
import random
from matplotlib import pyplot as plt
import cv2

def resize_image(img, target_size=(256, 256)):
    return cv2.resize(img.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)

def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color='c', rotate_90=True):
    """disp shape (2, H, W)"""

    if rotate_90:
        # Swap H and W coordinates and adjust displacement
        disp = np.array([disp[1], -disp[0]])
        disp = np.rot90(disp, k=1, axes=(1, 2))
        if bg_img is not None:
            bg_img = np.rot90(bg_img, k=1)

    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(
                                           0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **
                kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **
                kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize, color='white')
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def plot_result_fig(vis_data_dict, metrics_dict, save_path=None, title_font_size=14, dpi=300, show=False, close=False):
    """Plot visual results in a single figure/subplots.
    Images should be shaped (*sizes)
    Disp should be shaped (ndim, *sizes)

    vis_data_dict.keys() = ['fixed', 'moving', 'fixed_original',
                            'fixed_pred', 'warped_moving',
                            'disp_gt', 'disp_pred', 'mask']
    """

    print(np.unique(vis_data_dict["mask"]))

    max_size = max(vis_data_dict["moving"].shape)

    # Apply to all images
    for key in ['fixed', 'moving', 'warped', 'disp_pred', 'mask']:
        vis_data_dict[key] = resize_image(vis_data_dict[key], (max_size,max_size))
   
    fig = plt.figure(figsize=(24, 6))
    # Set the figure's background color to black
    fig.set_facecolor('black')
    title_pad = 8
    plot_n = 6

    ax = plt.subplot(1, plot_n, 1)
    plt.imshow(vis_data_dict["fixed"], cmap='gray')
    plt.axis('off')
    ax.set_title('Fixed', fontsize=title_font_size, pad=title_pad, color='white')

    ax = plt.subplot(1, plot_n, 2)
    plt.imshow(vis_data_dict["moving"], cmap='gray')
    plt.axis('off')
    ax.set_title('Moving', fontsize=title_font_size, pad=title_pad, color='white')

    # calculate the error before and after reg
    error_before = vis_data_dict["fixed"] - vis_data_dict["moving"]
    error_after = vis_data_dict["fixed"] - vis_data_dict["warped"]

    # Apply mask if available - use 0 (black) instead of np.nan
    if vis_data_dict["mask"] is not None:
        # Convert mask to boolean if it isn't already
        mask_bool = vis_data_dict["mask"].astype(bool)
        
        error_before_masked = np.where(mask_bool, error_before, 0)
        error_after_masked = np.where(mask_bool, error_after, 0)
        
        # Calculate dynamic min/max only from valid (masked) regions
        valid_before = error_before[mask_bool]
        valid_after = error_after[mask_bool]
        errors_combined = np.concatenate([valid_before.flatten(), valid_after.flatten()])
    else:
        error_before_masked = error_before
        error_after_masked = error_after
        errors_combined = np.concatenate([error_before.flatten(), error_after.flatten()])

    # Calculate dynamic min/max for consistent scaling
    if len(errors_combined) > 0:
        vmin = np.percentile(errors_combined, 5)
        vmax = np.percentile(errors_combined, 95)
        v_abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -v_abs_max, v_abs_max
    else:
        vmin, vmax = -2, 2  # fallback

    # error before
    ax = plt.subplot(1, plot_n, 4)
    # Set axis background to black
    ax.set_facecolor('black')
    im_before = plt.imshow(error_before_masked, vmin=vmin, vmax=vmax, cmap='seismic')
    # Make masked regions transparent so black background shows through
    im_before.set_array(np.ma.masked_where(~mask_bool, error_before_masked))
    plt.axis('off')
    ax.set_title(f'TRE (before): {metrics_dict["tre_mean_before"].item():.2f} ± {metrics_dict["tre_std_before"].item():.2f}', fontsize=title_font_size, pad=title_pad, color='white')

    # error after
    ax = plt.subplot(1, plot_n, 5)
    # Set axis background to black
    ax.set_facecolor('black')
    im_after = plt.imshow(error_after_masked, vmin=vmin, vmax=vmax, cmap='seismic')
    # Make masked regions transparent so black background shows through
    im_after.set_array(np.ma.masked_where(~mask_bool, error_after_masked))
    plt.axis('off')
    ax.set_title(f'TRE (after): {metrics_dict["tre_mean_after"].item():.2f} ± {metrics_dict["tre_std_after"].item():.2f}', fontsize=title_font_size, pad=title_pad, color='white')
        # Replace the error map section in your code with this:

       # predicted fixed image
    ax = plt.subplot(1, plot_n, 3)
    plt.imshow(vis_data_dict["warped"], cmap='gray')
    plt.axis('off')
    ax.set_title('Warped', fontsize=title_font_size, pad=title_pad, color="white")

    ax = plt.subplot(1, plot_n, 6)
    bg_img = np.zeros_like(vis_data_dict["fixed"])

    plot_warped_grid(ax, vis_data_dict["disp_pred"], bg_img,
                 interval=3, title=f"$\phi_{{pred}}$={metrics_dict['folding_ratio'].item()*100:.1f}%", fontsize=title_font_size)

    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9,
                        bottom=0.1, wspace=0.001, hspace=0.1)

    # saving
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    if close:
        plt.close()
    return fig



def visualise_result(data_dict, metrics_dict, axis=0, save_result_dir=None, epoch=None, dpi=100, id=None, show=False, close=False, hyper=0):
    """
    Save one validation visualisation figure for each epoch.
    - 2D: 1 random slice from N-slice stack (not a sequence)
    - 3D: the middle slice on the chosen axis

    Args:
        data_dict: (dict) images shape (N, 1, *sizes), disp shape (N, ndim, *sizes)
        save_result_dir: (string) Path to visualisation result directory
        epoch: (int) Epoch number (for naming when saving)
        axis: (int) For 3D only, choose the 2D plane orthogonal to this axis in 3D volume
        dpi: (int) Image resolution of saved figure
    """
    # check cast to Numpy array
    for n, d in data_dict.items():
        if isinstance(d, torch.Tensor):
            data_dict[n] = d.detach().cpu().numpy()

    # print(data_dict.items())

    ndim = data_dict["fixed"].ndim - 2
    sizes = data_dict["fixed"].shape[2:]

    # put 2D slices into visualisation data dict
    vis_data_dict = {}
    if ndim == 2:
        # randomly choose a slice for 2D
        z = random.randint(0, data_dict["fixed"].shape[0]-1)
        for name, d in data_dict.items():
            vis_data_dict[name] = data_dict[name][z, ...].squeeze()

    else:  # 3D
        # visualise the middle slice of the chosen axis
        z = int(sizes[axis] // 2)
        for name, d in data_dict.items():
            if name in ["disp_pred", "disp_gt", "disp"]:
                # dvf.yaml: choose the two axes/directions to visualise
                axes = [0, 1, 2]
                axes.remove(axis)
                vis_data_dict[name] = d[0, axes, ...].take(
                    z, axis=axis+1)  # (2, X, X)
            elif name in ["fixed", "moving", "warped", "fixed_seg", "moving_seg", "warped_seg", "mask"]:
                # images and mask - added "mask" to this list
                vis_data_dict[name] = d[0, 0, ...].take(z, axis=axis)  # (X, X)
            else:
                # Handle any other keys that might exist (like mask with different structure)
                # This catches mask if it has a different dimensionality
                if name == "mask":
                    if d.ndim == len(sizes):  # mask has shape (*sizes)
                        vis_data_dict[name] = d.take(z, axis=axis)
                    elif d.ndim == len(sizes) + 1:  # mask has shape (1, *sizes) or (N, *sizes)
                        vis_data_dict[name] = d[0, ...].take(z, axis=axis)
                    elif d.ndim == len(sizes) + 2:  # mask has shape (N, 1, *sizes)
                        vis_data_dict[name] = d[0, 0, ...].take(z, axis=axis)
                else:
                    # For any other unknown keys, try to extract slice
                    try:
                        if d.ndim >= len(sizes):
                            vis_data_dict[name] = d[0, 0, ...].take(z, axis=axis) if d.ndim == len(sizes) + 2 else d[0, ...].take(z, axis=axis)
                    except:
                        # If slicing fails, skip this key
                        pass

    # housekeeping: dummy dvf_gt for inter-subject case
    if not "disp_gt" in data_dict.keys():
        vis_data_dict["disp_gt"] = np.zeros_like(vis_data_dict["disp_pred"])

    # set up figure saving path
    if save_result_dir is not None:
        if id is not None:
            fig_save_path = os.path.join(
                save_result_dir, f'epoch{epoch}_axis_{axis}_silce_{z}_id_{id}_{hyper}.png')
        else:
            fig_save_path = os.path.join(
                save_result_dir, f'epoch{epoch}_axis_{axis}_slice_{z}.png')
    else:
        fig_save_path = None

    fig = plot_result_fig(
        vis_data_dict, save_path=fig_save_path, dpi=dpi, show=show, close=close, metrics_dict=metrics_dict)
    return fig
