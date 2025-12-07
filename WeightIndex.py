"""
Real-time Semantic Segmentation Evaluation Metric

This script implements the weighted evaluation metric proposed to balance
speed (FPS) and accuracy (mIoU) for real-time semantic segmentation models.

Mathematical Formulation:
-------------------------
The evaluation is based on Min-Max normalization and weighted sum:

1. Normalized Accuracy (mIoU):
   mIoU_hat = (mIoU_i - mIoU_min) / (mIoU_max - mIoU_min)

2. Normalized Speed (FPS):
   Speed_hat = (Speed_i - Speed_min) / (Speed_max - Speed_min)

3. Weighted Score:
   Score = alpha * mIoU_hat + beta * Speed_hat

Default settings: alpha = 0.5, beta = 0.5
"""

import math

def calculate_weighted_score(models_data, alpha=0.5, beta=0.5):
    """
    Calculates the weighted score for a list of model performance data.

    Args:
        models_data (list of dict): List containing model info.
                                    Each dict must have 'mIoU' and 'FPS' keys.
        alpha (float): Weight for accuracy (mIoU). Default is 0.5.
        beta (float): Weight for speed (FPS). Default is 0.5.

    Returns:
        list of dict: The original data with an added 'Score' and 'Score_Pct' key,
                      sorted by Score in descending order.
    """

    # Extract lists to find global min/max
    miou_list = [item['mIoU'] for item in models_data]
    fps_list = [item['FPS'] for item in models_data]

    # Calculate global Min and Max values
    miou_min, miou_max = min(miou_list), max(miou_list)
    fps_min, fps_max = min(fps_list), max(fps_list)

    print(f"{'-'*60}")
    print(f"Statistics for Normalization:")
    print(f"mIoU Range: [{miou_min}, {miou_max}]")
    print(f"FPS  Range: [{fps_min}, {fps_max}]")
    print(f"Weights   : Alpha={alpha}, Beta={beta}")
    print(f"{'-'*60}\n")

    # Calculate scores for each model
    scored_models = []
    for model in models_data:
        # Get raw values
        m = model['mIoU']
        f = model['FPS']

        # Normalize mIoU
        # Avoid division by zero if max == min
        m_hat = (m - miou_min) / (miou_max - miou_min) if miou_max > miou_min else 0.0

        # Normalize Speed (FPS)
        f_hat = (f - fps_min) / (fps_max - fps_min) if fps_max > fps_min else 0.0

        # Calculate Weighted Score
        score = (alpha * m_hat) + (beta * f_hat)

        # Append result (keeping original data)
        result_entry = model.copy()
        result_entry['Norm_mIoU'] = m_hat
        result_entry['Norm_FPS'] = f_hat
        result_entry['Score'] = score
        result_entry['Score_Pct'] = score * 100  # Convert to percentage for display

        scored_models.append(result_entry)

    # Sort by Score (Descending)
    scored_models.sort(key=lambda x: x['Score'], reverse=True)

    return scored_models

def main():
    # ==========================================
    # 1. Input Data
    # ==========================================
    # Format: {'id': 'Model Name', 'mIoU': value, 'FPS': value}
    data = [
        {'id': '1',  'mIoU': 77.1, 'FPS': 41.0},
        {'id': '2',  'mIoU': 78.1, 'FPS': 37.0},
        {'id': '3',  'mIoU': 74.8, 'FPS': 67.2},
        {'id': '4',  'mIoU': 75.3, 'FPS': 56.3},
        {'id': '5',  'mIoU': 75.4, 'FPS': 39.9},
        {'id': '6',  'mIoU': 76.5, 'FPS': 18.4},
        {'id': '7',  'mIoU': 75.9, 'FPS': 76.5},
        {'id': '8',  'mIoU': 74.7, 'FPS': 65.5},
        {'id': '9',  'mIoU': 75.3, 'FPS': 47.3},
        {'id': '10', 'mIoU': 75.3, 'FPS': 52.5},
        {'id': '11', 'mIoU': 76.0, 'FPS': 36.3},
        {'id': '12', 'mIoU': 75.3, 'FPS': 74.8},
        {'id': '13', 'mIoU': 76.8, 'FPS': 58.2},
        {'id': '14', 'mIoU': 74.9, 'FPS': 96.0},
        {'id': '15', 'mIoU': 77.5, 'FPS': 68.2},
        {'id': '16', 'mIoU': 75.8, 'FPS': 59.1},
        {'id': '17', 'mIoU': 78.1, 'FPS': 45.7},
        {'id': '18', 'mIoU': 77.8, 'FPS': 87.6},
        {'id': '19', 'mIoU': 78.9, 'FPS': 30.4},
        {'id': '20', 'mIoU': 77.4, 'FPS': 146.7},
        {'id': '21', 'mIoU': 77.7, 'FPS': 54.5},
        {'id': '22', 'mIoU': 79.4, 'FPS': 54.5},
        {'id': '23', 'mIoU': 80.4, 'FPS': 31.4},
        {'id': '24', 'mIoU': 76.2, 'FPS': 125.1},
        {'id': '25', 'mIoU': 78.4, 'FPS': 125.1},
        {'id': '26', 'mIoU': 80.1, 'FPS': 48.5},
        {'id': '27', 'mIoU': 80.6, 'FPS': 35.5},
        {'id': '28', 'mIoU': 77.4, 'FPS': 162.1},
        {'id': '29', 'mIoU': 78.3, 'FPS': 64.2},
        {'id': '30', 'mIoU': 78.8, 'FPS': 48.2},
        {'id': '31', 'mIoU': 78.7, 'FPS': 162.1},
        {'id': '32', 'mIoU': 79.8, 'FPS': 64.2},
        {'id': '33', 'mIoU': 80.3, 'FPS': 48.2}
    ]

    # ==========================================
    # 2. Perform Calculation
    # ==========================================
    # Default: alpha=0.5 (Accuracy), beta=0.5 (Speed)
    results = calculate_weighted_score(data, alpha=0.5, beta=0.5)

    # ==========================================
    # 3. Display Results
    # ==========================================
    # Print Header
    header = f"{'Rank':<5} | {'Row ID':<6} | {'mIoU':<6} | {'FPS':<6} | {'Index (%)':<10}"
    print(header)
    print("-" * len(header))

    # Print Rows
    for rank, row in enumerate(results):
        print(f"{rank+1:<5} | {row['id']:<6} | {row['mIoU']:<6.1f} | {row['FPS']:<6.1f} | {row['Score_Pct']:<10.2f}")

    # Identify the best model
    best_model = results[0]
    print("\n" + "="*40)
    print(f"Best Model: Row {best_model['id']}")
    print(f"Score: {best_model['Score_Pct']:.2f}% (mIoU: {best_model['mIoU']}, FPS: {best_model['FPS']})")
    print("="*40)

if __name__ == "__main__":
    main()