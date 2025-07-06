import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("TkAgg")

# y_true_rain = np.array([0, 0, 1])
# y_true_fog = np.array([2, 0, 3])
# y_true_a = np.column_stack([y_true_rain, y_true_fog])
#
# y_pred_rain = np.array([0, 0, 2])
# y_pred_fog = np.array([1, 0, 0])
# y_pred_a = np.column_stack([y_pred_rain, y_pred_fog])

# a) First example data
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



# a) Confusion matrix for Rain
cm_rain = confusion_matrix(y_true_a[:, 0], y_pred_a[:, 0])

# b) Confusion matrix for Fog
cm_fog = confusion_matrix(y_true_a[:, 1], y_pred_a[:, 1])

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

# b) alternative Error categorization
# Define declarative error rules
error_rules = [
    {
        "name": "False Alarm Both",
        "condition": lambda t, p: t[0] == 0 and t[1] == 0 and (p[0] > 0 or p[1] > 0)
    },
    {
        "name": "Cross: Fog→Rain",
        "condition": lambda t, p: t[0] == 0 and t[1] > 0 and p[0] > 0
    },
    {
        "name": "Cross: Rain→Fog",
        "condition": lambda t, p: t[0] > 0 and t[1] == 0 and p[1] > 0
    },
    {
        "name": "Overestimate",
        "condition": lambda t, p: p[0] > t[0] or p[1] > t[1]
    },
    {
        "name": "Underestimate",
        "condition": lambda t, p: p[0] < t[0] or p[1] < t[1]
    },
    {
        "name": "Correct",
        "condition": lambda t, p: tuple(p) == tuple(t)
    }
]

# Categorization function
def categorize_error_table(y_true, y_pred, rules):
    categories = []
    for yt, yp in zip(y_true, y_pred):
        for rule in rules:
            if rule["condition"](yt, yp):
                categories.append(rule["name"])
                break
    return categories

# Apply rules to data
categories = categorize_error_table(y_true_a, y_pred_a, error_rules)

def build_error_table(y_true, y_pred, rules):
    rows = []
    for yt, yp in zip(y_true, y_pred):
        for rule in rules:
            if rule["condition"](yt, yp):
                rows.append({
                    "True_Rain": yt[0],
                    "True_Fog": yt[1],
                    "Pred_Rain": yp[0],
                    "Pred_Fog": yp[1],
                    "Error_Type": rule["name"]
                })
                break
    return pd.DataFrame(rows)
error_table = build_error_table(y_true_a, y_pred_a, error_rules)
print(error_table)


# b) Error categorization table
def categorize_error_a(yt, yp):
    t_r, t_f = yt
    p_r, p_f = yp
    if t_r == 0 and t_f == 0 and (p_r > 0 or p_f > 0):
        return "False Alarm Both"
    elif t_f > 0 and t_r == 0 and p_r > 0:
        return "Cross: Fog→Rain"
    elif t_r > 0 and t_f == 0 and p_f > 0:
        return "Cross: Rain→Fog"
    elif p_r > t_r or p_f > t_f:
        return "Overestimate"
    elif p_r < t_r or p_f < t_f:
        return "Underestimate"
    else:
        return "Correct"

error_types_a = [categorize_error_a(yt, yp) for yt, yp in zip(y_true_a, y_pred_a)]

error_table_a = pd.DataFrame({
    "True_Rain": y_true_a[:, 0],
    "True_Fog": y_true_a[:, 1],
    "Pred_Rain": y_pred_a[:, 0],
    "Pred_Fog": y_pred_a[:, 1],
    "Error_Type": error_types_a
})

# c) Cross-label heatmap
def label_status(rain, fog):
    if rain > 0 and fog > 0:
        return "Rain+Fog"
    elif rain > 0:
        return "Rain"
    elif fog > 0:
        return "Fog"
    else:
        return "None"

true_status_a = [label_status(r, f) for r, f in y_true_a]
pred_status_a = [label_status(r, f) for r, f in y_pred_a]

cross_matrix_a = pd.crosstab(pd.Series(true_status_a, name="True"),
                             pd.Series(pred_status_a, name="Predicted"))

# d) Combined confusion matrix using joint labeling
y_true_joint_a = [f"{r},{f}" for r, f in y_true_a]
y_pred_joint_a = [f"{r},{f}" for r, f in y_pred_a]
unique_labels_joint = sorted(set(y_true_joint_a + y_pred_joint_a))
conf_matrix_joint = pd.crosstab(pd.Series(y_true_joint_a, name="True"),
                                pd.Series(y_pred_joint_a, name="Predicted"))

# Show error categorization table
#import ace_tools as tools; tools.display_dataframe_to_user(name="(b) Error Categorization Table (Example A)", dataframe=error_table_a)

# Show cross-label heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cross_matrix_a, annot=True, fmt="d", cmap="Blues")
plt.title("(c) Cross-Label Confusion Heatmap")
plt.tight_layout()
plt.show()

# Show combined joint confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_joint, annot=True, fmt="d", cmap="Greens")
plt.title("(d) Combined Confusion Matrix (Rain,Fog pairs)")
plt.tight_layout()
plt.show()
