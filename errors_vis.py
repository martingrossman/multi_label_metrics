import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")


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

# -----------------------------
# (c) Cross-Label Confusion Matrix
# -----------------------------
def label_status(rain, fog):
    if rain > 0 and fog > 0:
        return "Rain+Fog"
    elif rain > 0:
        return "Rain"
    elif fog > 0:
        return "Fog"
    else:
        return "None"

true_status = [label_status(r, f) for r, f in y_true_a]
pred_status = [label_status(r, f) for r, f in y_pred_a]

cross_matrix = pd.crosstab(pd.Series(true_status, name="True"),
                           pd.Series(pred_status, name="Predicted"))

# -----------------------------
# (d) Combined Confusion Matrix (Rain, Fog pairs)
# -----------------------------
y_true_joint = [(r, f) for r, f in y_true_a]
y_pred_joint = [(r, f) for r, f in y_pred_a]

conf_matrix_joint = pd.crosstab(pd.Series(y_true_joint, name="True"),
                                pd.Series(y_pred_joint, name="Predicted"))

# -----------------------------
# Visualization
# -----------------------------

# Show error categorization table (if in notebook or interactive)
#import ace_tools as tools
#tools.display_dataframe_to_user(name="(b) Error Categorization Table", dataframe=error_table)

# (c) Cross-label heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cross_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("(c) Cross-Label Confusion Heatmap")
plt.tight_layout()
plt.show()

# (d) Combined (Rain, Fog) label heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_joint, annot=True, fmt="d", cmap="Greens")
plt.title("(d) Combined Confusion Matrix (Rain,Fog Pairs)")
plt.tight_layout()
plt.show()