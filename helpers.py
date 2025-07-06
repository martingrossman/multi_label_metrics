import numpy as np


def create_data():
    y_true_a = np.array([
        [0, 2],  # Sample 1: Clear rain, moderate fog
        [0, 0],  # Sample 2: Clear
        [1, 3]  # Sample 3: Mild rain, severe fog
    ])
    y_pred_a = np.array([
        [0, 1],  # Slight under on fog
        [0, 0],  # Correct
        [2, 0]  # Over rain, miss fog
    ])

    return y_true_a, y_pred_a
