import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("TkAgg")

GENERATE_RAND_GT = 1

if GENERATE_RAND_GT:
    # Set random seed for reproducibility
    np.random.seed(42)

    # Simulate a random but structurally similar version of the data
    # We want 3 samples like the original shape, with values 0–3 for fog/rain

    y_true_rain = np.random.choice([0, 1, 2], size=3)
    y_true_fog = np.random.choice([0, 1, 2, 3], size=3)
    y_true_a = np.column_stack([y_true_rain, y_true_fog])

    y_pred_rain = np.random.choice([0, 1, 2], size=3)
    y_pred_fog = np.random.choice([0, 1, 2, 3], size=3)
    y_pred_a = np.column_stack([y_pred_rain, y_pred_fog])

    y_true_a, y_pred_a
else:
    y_true_a = np.array([
        [0, 2],   # Sample 1: Clear rain, moderate fog
        [0, 0],   # Sample 2: Clear
        [1, 3]    # Sample 3: Mild rain, severe fog
    ])
    y_pred_a = np.array([
        [0, 1],   # Slight under on fog
        [0, 0],   # Correct
        [2, 0]    # Over rain, miss fog
    ])

print(y_true_a)
print(y_pred_a)

# a) Confusion matrix for Rain
cm_rain = confusion_matrix(y_true_a[:, 0], y_pred_a[:, 0],labels=[0, 1, 2])

# b) Confusion matrix for Fog
cm_fog = confusion_matrix(y_true_a[:, 1], y_pred_a[:, 1],labels=[0, 1, 2, 3])

# c) Confusion matrix for Rain+Fog
# Convert tuple labels to string labels for confusion matrix computation
y_true_labels = [f"{r},{f}" for (r, f) in y_true_a]
y_pred_labels = [f"{r},{f}" for (r, f) in y_pred_a]

# Define all unique classes (sorted) for consistent axis ordering
unique_labels = sorted(set(y_true_labels + y_pred_labels))

# Compute confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)

# Compute confusion matrix
cm_joint = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)


# Plotting
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.heatmap(cm_rain, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Rain Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(cm_fog, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Fog Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")


sns.heatmap(cm_joint, annot=True, fmt='d', cmap='Purples',
            xticklabels=unique_labels, yticklabels=unique_labels,ax=axes[2])
axes[2].set_title("Joint Rain+Fog Confusion Matrix (Tuples as Labels)")
axes[2].set_xlabel("Predicted [Rain,Fog]")
axes[2].set_ylabel("True [Rain,Fog]")

plt.tight_layout()
plt.show()