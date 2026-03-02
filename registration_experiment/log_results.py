import csv

def save_metrics_to_csv(case_id, accuracy_mean_before, accuracy_std_before, accuracy_mean_after, accuracy_std_after,
                        ncc_before, ncc_after, folding_ratio, jacobian_det, metrics, output_dir="metrics_output.csv"):
    """
    Saves the metrics data to a CSV file.
    
    :param case_id: The case identifier.
    :param accuracy_mean_before: TRE mean before.
    :param accuracy_std_before: TRE standard deviation before.
    :param accuracy_mean_after: TRE mean after.
    :param accuracy_std_after: TRE standard deviation after.
    :param ncc_before: NCC score before the transformation.
    :param ncc_after: NCC score after the transformation.
    :param folding_ratio: Folding ratio.
    :param jacobian_det: Jacobian determinant values.
    :param metrics: Dictionary holding additional metrics to save.
    :param output_dir: Path to the CSV file.
    """
    # Check if file exists, and if not, write headers
    file_exists = False
    try:
        with open(output_dir, mode='r') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    
    # Open the CSV file in append mode
    with open(output_dir, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writerow([
                'Case ID', 'TRE Mean Before', 'TRE Std Before', 'TRE Mean After', 'TRE Std After',
                'NCC Before Warp', 'NCC After Warp', 'Folding Ratio', 'Min Jacobian', 'Mean Jacobian', 'Max Jacobian'
            ])
        
        # Write the metrics for the current case
        writer.writerow([
            case_id,
            accuracy_mean_before[0], accuracy_std_before[0],  # TRE mean and std before
            accuracy_mean_after[0], accuracy_std_after[0],  # TRE mean and std after
            ncc_before,   # NCC before warp
            ncc_after,    # NCC after warp
            folding_ratio,  # Folding ratio
            jacobian_det.min().item(),  # Min Jacobian determinant
            jacobian_det.mean().item(),  # Mean Jacobian determinant
            jacobian_det.max().item()   # Max Jacobian determinant
        ])

    print(f"Metrics for case {case_id} saved to {output_dir}")
