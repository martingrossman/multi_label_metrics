import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("TkAgg")



# Updated version of the function with ✓/✖ colored (✓ in green, ✖ in red)

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

def plot_annotated_cross_confusion_matrix_colored(y_true, y_pred, title="Cross-Label Confusion Matrix"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.colors import to_hex

    # 1. Derive situation labels
    def label_status(rain, fog):
        if rain > 0 and fog > 0:
            return "Rain+Fog"
        elif rain > 0:
            return "Rain"
        elif fog > 0:
            return "Fog"
        else:
            return "None"

    true_status = [label_status(r, f) for r, f in y_true]
    pred_status = [label_status(r, f) for r, f in y_pred]

    # 2. Confusion matrix
    cross_matrix = pd.crosstab(
        pd.Series(true_status, name="True"),
        pd.Series(pred_status, name="Predicted")
    )

    # 3. Annotated matrix with color codes for ✓ and ✖
    annot_matrix = cross_matrix.copy().astype(str)
    text_colors = {}

    for i, true_val in enumerate(cross_matrix.index):
        for j, pred_val in enumerate(cross_matrix.columns):
            count = cross_matrix.iloc[i, j]
            if count == 0:
                annot_matrix.iloc[i, j] = ""
                text_colors[(i, j)] = 'black'
            else:
                indices = [
                    k for k in range(len(y_true))
                    if true_status[k] == true_val and pred_status[k] == pred_val
                ]
                rain_errors = [abs(y_true[k][0] - y_pred[k][0]) for k in indices]
                fog_errors = [abs(y_true[k][1] - y_pred[k][1]) for k in indices]
                avg_rain_err = np.mean(rain_errors)
                avg_fog_err = np.mean(fog_errors)
                true_raws = [tuple(y_true[k]) for k in indices]
                pred_raws = [tuple(y_pred[k]) for k in indices]
                all_match = all(tr == pr for tr, pr in zip(true_raws, pred_raws))
                if all_match:
                    symbol = "✓"
                    text_colors[(i, j)] = "green"
                else:
                    symbol = "✖"
                    text_colors[(i, j)] = "red"
                annot_matrix.iloc[i, j] = f"{count}\n{symbol} (R:{avg_rain_err:.1f}, F:{avg_fog_err:.1f})"

    # 4. Plot with text colors
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cross_matrix, annot=False, fmt="", cmap="Blues", cbar=False,
                linewidths=0.5, linecolor='gray', square=True, ax=ax)

    for i in range(len(cross_matrix.index)):
        for j in range(len(cross_matrix.columns)):
            text = annot_matrix.iloc[i, j]
            if text:
                ax.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center',
                        color=text_colors[(i, j)],
                        fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_xticklabels(cross_matrix.columns, rotation=45)
    ax.set_yticklabels(cross_matrix.index, rotation=0)
    plt.tight_layout()
    plt.show()

# Call the new version on the original sample
plot_annotated_cross_confusion_matrix_colored(y_true_a, y_pred_a, title="(Colored) Enhanced Confusion Matrix with ✓/✖")
