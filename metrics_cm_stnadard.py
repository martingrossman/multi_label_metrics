import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("TkAgg")

# -------------------------------
# Simulate input (as in your code)
# -------------------------------
GENERATE_RAND_GT = 1

if GENERATE_RAND_GT:
    np.random.seed(42)
    N = 100
    y_true_rain = np.random.choice([0, 1, 2], size=N)
    y_true_fog = np.random.choice([0, 1, 2, 3], size=N)
    y_true_a = np.column_stack([y_true_rain, y_true_fog])

    y_pred_rain = np.random.choice([0, 1, 2], size=N)
    y_pred_fog = np.random.choice([0, 1, 2, 3], size=N)
    y_pred_a = np.column_stack([y_pred_rain, y_pred_fog])

    y_pred_a = y_true_a
else:
    y_true_a = np.array([[0, 2], [0, 0], [1, 3]])
    y_pred_a = np.array([[0, 1], [0, 0], [2, 0]])

print("y_true_a =\n", y_true_a)
print("y_pred_a =\n", y_pred_a)

# -----------------------------------
# a) Confusion matrix for Rain
# b) Confusion matrix for Fog
# -----------------------------------
cm_rain = confusion_matrix(y_true_a[:, 0], y_pred_a[:, 0], labels=[0, 1, 2])
cm_fog = confusion_matrix(y_true_a[:, 1], y_pred_a[:, 1], labels=[0, 1, 2, 3])

# -----------------------------------
# c) Confusion matrix for Rain+Fog (only in dataset)
# d) Full confusion matrix for all combinations
# -----------------------------------
y_true_labels = [f"{r},{f}" for (r, f) in y_true_a]
y_pred_labels = [f"{r},{f}" for (r, f) in y_pred_a]
unique_labels = sorted(set(y_true_labels + y_pred_labels))
cm_joint = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)

# (d) Full 3x4 combinations for all Rain in [0,1,2], Fog in [0,1,2,3]
from itertools import product
all_possible_labels = [f"{r},{f}" for r, f in product(range(3), range(4))]

cm_full_joint = confusion_matrix(
    y_true_labels, y_pred_labels, labels=all_possible_labels
)

# -----------------------------------
# Plotting all 4 confusion matrices
# -----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.heatmap(cm_rain, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title("Rain Confusion Matrix")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("True")

sns.heatmap(cm_fog, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
axes[0, 1].set_title("Fog Confusion Matrix")
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("True")

sns.heatmap(cm_joint, annot=True, fmt='d', cmap='Purples',
            xticklabels=unique_labels, yticklabels=unique_labels, ax=axes[1, 0])
axes[1, 0].set_title("Joint Rain+Fog Matrix (Only Present Combos)")
axes[1, 0].set_xlabel("Predicted [Rain,Fog]")
axes[1, 0].set_ylabel("True [Rain,Fog]")

sns.heatmap(cm_full_joint, annot=True, fmt='d', cmap='Oranges',
            xticklabels=all_possible_labels, yticklabels=all_possible_labels, ax=axes[1, 1])
axes[1, 1].set_title("Full Joint Matrix (All Rain-Fog Combos)")
axes[1, 1].set_xlabel("Predicted [Rain,Fog]")
axes[1, 1].set_ylabel("True [Rain,Fog]")

plt.tight_layout()
plt.show()
