import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from image_shift import augment_dataset
from load_split_data import load_split_data


def plot_original_vs_reconstructed(X_original, X_reconstructed, n=10):
    """
    Plot n original and reconstructed images side by side.
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
    plt.show()


def interactive_pca_reconstruction(X_data, sample_index=0, pca_components=68):
    """
    Create an interactive slider to show reconstructed images as PCA components increase.
    """
    original = X_data[sample_index]

    def update(n_components):

        # Recreate PCA with `n_components`
        temp_pca = PCA(n_components=n_components)
        X_pca = temp_pca.fit_transform(X_data)
        X_recon = temp_pca.inverse_transform(X_pca)

        # Plot original vs reconstructed
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(original.reshape(28, 28), cmap="gray")
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(X_recon[sample_index].reshape(28, 28), cmap="gray")
        axs[1].set_title(f"Reconstructed\n{n_components} components")
        axs[1].axis("off")
        plt.show()

    # Slider from 1 to full 784 features
    slider = widgets.IntSlider(min=1, max=200, step=5, value=pca_components)
    display(widgets.interact(update, n_components=slider))


def pca_mnist_dataset_analysis():
    # 1. Load MNIST dataset

    X_train, X_test, y_train, y_test = load_split_data()

    # Apply image shift data augmentation
    print("⚙️ Applying image shift data augmentation...")
    X_train, y_train = augment_dataset(X_train, y_train)

    n_components = 0.95

    # 2. Apply PCA to retain 75% variance
    print("⚙️ Applying PCA (n_components=0.95)...")
    pca = PCA(n_components=n_components, svd_solver="auto")
    X_train_pca = pca.fit_transform(X_train)

    # Show slider for sample 0
    interactive_pca_reconstruction(
        X_train, sample_index=0, pca_components=pca.n_components_
    )

    X_test_pca = pca.transform(X_test)

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
        # "mnist_pca_95_data.npz",
        save_path=f"/content/drive/MyDrive/mnist_pca_{n_components}_data.npz",
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        y_train=y_train,
        y_test=y_test,
    )

    plot_original_vs_reconstructed(X_train, X_train_reconstructed, n=10)


if __name__ == "__main__":
    pca_mnist_dataset_analysis()


# def simulate_pca_component_effect(X_data, sample_index=0, steps=[5, 10, 20, 30, 50, 100]):
#     original = X_data[sample_index]

#     for n_components in steps:
#         temp_pca = PCA(n_components=n_components)
#         X_pca = temp_pca.fit_transform(X_data)
#         X_recon = temp_pca.inverse_transform(X_pca)

#         fig, axs = plt.subplots(1, 2, figsize=(6, 3))
#         axs[0].imshow(original.reshape(28, 28), cmap='gray')
#         axs[0].set_title("Original")
#         axs[0].axis('off')

#         axs[1].imshow(X_recon[sample_index].reshape(28, 28), cmap='gray')
#         axs[1].set_title(f"Reconstructed\n{n_components} comps")
#         axs[1].axis('off')

#         plt.tight_layout()
#         plt.savefig(f"reconstructed_{n_components}_components.png")
#         plt.show()


# simulate_pca_component_effect(X_train, sample_index=0)
