# To plot pretty figures
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# Set Matplotlib defaults
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


# Function to save figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# === After fitting the SVR model ===
def analyze_support_vectors(model):
    print("Number of support vectors:", len(model.support_))
    print("Support vectors shape:", model.support_vectors_.shape)

    # For visualization, use only first 2 dimensions (for interpretability)
    if model.support_vectors_.shape[1] >= 2:
        support_vectors_2d = model.support_vectors_[:, :2]
    else:
        support_vectors_2d = model.support_vectors_

    plt.figure(figsize=(8, 6))
    plt.scatter(
        support_vectors_2d[:, 0],
        support_vectors_2d[:, 1],
        c="blue",
        s=30,
        edgecolors="k",
        label="Support Vectors",
        alpha=0.6,
    )

    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.title("Support Vectors (First 2 Features)")
    plt.legend()

    # Save the plot
    save_fig("support_vectors_plot")
    plt.show()
