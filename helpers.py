import numpy as np
from sklearn.metrics import confusion_matrix


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


def get_cm_ith_cat(i, y_true_a, y_pred_a):
    cm_i = confusion_matrix(y_true_a[:, i], y_pred_a[:, i])
    return cm_i
