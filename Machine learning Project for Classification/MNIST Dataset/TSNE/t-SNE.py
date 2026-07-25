import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Load MNIST
print("Loading MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32)
y = y.astype(int)

# Optional: Subsample for speed (t-SNE is slow)
sample_size = 6000
indices = np.random.choice(len(X), size=sample_size, replace=False)
X_sample = X[indices]
y_sample = y[indices]

# Step 2: Standardize features
X_scaled = StandardScaler().fit_transform(X_sample)

# Step 3: Apply t-SNE
print("Applying t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_2d = tsne.fit_transform(X_scaled)

# Step 4: Plot
print("Plotting...")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Digit Label')
plt.title("t-SNE on MNIST (2D projection)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()
