import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from load_split_data import load_split_data


def plot_original_vs_reconstructed(X_original, X_reconstructed, n=10):
    """
    In the code, n=10 in the plot_original_vs_reconstructed function determines the number of original and reconstructed images that will be displayed side by side in the plot. The function will show 10 original images and their corresponding 10 reconstructed versions.
    """
    plt.figure(figsize=(n, 2))

    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_original[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("Original", fontsize=12)

        # Reconstructed
        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(X_reconstructed[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("Reconstructed", fontsize=12)

    plt.tight_layout()
    plt.savefig(fname="reconstruction_plot.png")
    plt.show()


def pca_mnist_dataset_with_no_data_augmentation_analysis():
    # 1. Load MNIST dataset

    X_train, X_test, y_train, y_test = load_split_data()

    n_components = 0.95

    # 2. Apply PCA to retain 95% variance
    print("⚙️ Applying PCA (n_components=0.95)...")
    pca = PCA(n_components=n_components, svd_solver="auto")
    X_train_pca = pca.fit_transform(X_train)

    X_test_pca = pca.transform(X_test)

    joblib.dump(pca, f"pca_train.joblib")
    print("\n✅ PCA for training is saved as pca_train.joblib")

    # 3. Explained variance ratio for each component
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\n✅ Total components selected: {pca.n_components_}")
    print("\n📊 Explained variance per component:")
    for i, var in enumerate(explained_var):
        print(f"Component {i+1:3}: {var:.4f} (Cumulative: {cumulative_var[i]:.4f})")

    # 4. Compute reconstruction loss (Mean Squared Error)
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    reconstruction_loss = mean_squared_error(X_train, X_train_reconstructed)
    print(f"\n🔁 Reconstruction loss (MSE) on training set: {reconstruction_loss:.4f}")

    # 5. Save compressed dataset
    print("\n💾 Saving PCA-reduced data as .npz...")
    np.savez_compressed(
        file=f"/content/drive/MyDrive/pca_{n_components}_with_no_data_augmentation_mnist_data.npz",
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        y_train=y_train,
        y_test=y_test,
    )

    plot_original_vs_reconstructed(X_train, X_train_reconstructed, n=10)


if __name__ == "__main__":
    pca_mnist_dataset_with_no_data_augmentation_analysis()
