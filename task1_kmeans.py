import numpy as np
import matplotlib.pyplot as plt
import sys
import time

def load_data(data_path, label_path):
    print(f"Loading data from {data_path}...")
    data = np.loadtxt(data_path, delimiter=',')
    print(f"Data shape: {data.shape}")
    
    print(f"Loading labels from {label_path}...")
    labels = np.loadtxt(label_path, delimiter=',')
    print(f"Labels shape: {labels.shape}")
    
    return data, labels

def euclidean_dist(X, centroids):
    # ||x - c||^2
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)**2

def cosine_dist(X, centroids):
    # 1 - cosine_similarity
    # Cosine Sim = (A . B) / (||A|| * ||B||)
    # X: (N, D), Centroids: (K, D)
    
    # Normalize X and Centroids
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    C_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Avoid division by zero
    X_norm[X_norm == 0] = 1e-10
    C_norm[C_norm == 0] = 1e-10
    
    sim = np.dot(X, centroids.T) / (X_norm @ C_norm.T)
    return 1 - sim

def jaccard_dist(X, centroids):
    # Generalized Jaccard = sum(min(A,B)) / sum(max(A,B))
    # Distance = 1 - J
    # X: (N, D), Centroids: (K, D)
    # This is computationally expensive to broadcast (N, K, D).
    # N=10000, K=10, D=784 -> 78.4M elements. Doable in memory.
    
    X_expand = X[:, np.newaxis, :] # (N, 1, D)
    C_expand = centroids[np.newaxis, :, :] # (1, K, D)
    
    numerator = np.sum(np.minimum(X_expand, C_expand), axis=2)
    denominator = np.sum(np.maximum(X_expand, C_expand), axis=2)
    
    # Avoid division by zero
    denominator[denominator == 0] = 1e-10
    
    sim = numerator / denominator
    return 1 - sim

def compute_sse(X, centroids, labels, metric='euclidean'):
    # SSE is sum of squared errors. For Euclidean, it's sum of squared distances.
    # For Cosine/Jaccard, "SSE" usually refers to the sum of distances (or squared distances?).
    # The question asks to "Compare the SSEs". Usually SSE implies squared Euclidean distance.
    # However, if we cluster with Cosine, minimizing Cosine distance is the objective.
    # Let's report the sum of squared distances for Euclidean, and sum of distances for others.
    
    if metric == 'euclidean':
        dists = euclidean_dist(X, centroids) # This returns squared distances already
    elif metric == 'cosine':
        dists = cosine_dist(X, centroids) # Distances
        dists = dists**2 # Squared? Usually SSE is squared error. Let's square it to be consistent with "SSE".
    elif metric == 'jaccard':
        dists = jaccard_dist(X, centroids)
        dists = dists**2
        
    # Select distance to assigned cluster
    sse = 0
    for i in range(len(X)):
        sse += dists[i, labels[i]]
        
    return sse

def kmeans(X, k, metric='euclidean', max_iters=100, tol=1e-4, stop_criteria='all'):
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    np.random.seed(42)
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    prev_sse = float('inf')
    
    for i in range(max_iters):
        # Compute distances
        if metric == 'euclidean':
            distances = euclidean_dist(X, centroids)
        elif metric == 'cosine':
            distances = cosine_dist(X, centroids)
        elif metric == 'jaccard':
            distances = jaccard_dist(X, centroids)
            
        labels = np.argmin(distances, axis=1)
        
        # Compute SSE for current assignment
        # Note: For SSE increase check, we need SSE of current centroids with current labels
        # But usually we update centroids then check.
        # Let's follow the standard: Assign -> Update -> Check
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                if metric == 'euclidean':
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    # For Cosine/Jaccard, mean is still a reasonable update rule for K-Means
                    # strictly speaking, K-Medoids or specific updates are better, but Mean is standard approximation.
                    new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j] # Keep old if empty
                
        # Calculate SSE with NEW centroids
        current_sse = compute_sse(X, new_centroids, labels, metric)
        
        # Check Stop Criteria
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        
        stop = False
        reasons = []
        
        # 1. No change in centroid position
        if centroid_shift < tol:
            stop = True
            reasons.append("centroid_shift")
            
        # 2. SSE value increases
        if current_sse > prev_sse:
            # If 'all' (OR condition), we stop.
            # If testing specific condition, we might ignore this.
            if stop_criteria == 'sse_increase' or stop_criteria == 'all':
                stop = True
                reasons.append("sse_increase")
        
        # 3. Max iterations (handled by loop, but let's be explicit)
        if i == max_iters - 1:
            stop = True
            reasons.append("max_iters")
            
        if stop_criteria == 'centroid_no_change' and centroid_shift < tol:
            break
        elif stop_criteria == 'sse_increase' and current_sse > prev_sse:
            break
        elif stop_criteria == 'max_iters' and i == max_iters - 1:
            break
        elif stop_criteria == 'all' and stop:
            break
            
        centroids = new_centroids
        prev_sse = current_sse
        
    return centroids, labels, i + 1, current_sse

def compute_purity(y_true, y_pred):
    contingency_matrix = np.zeros((10, 10))
    for i in range(len(y_true)):
        true_label = int(y_true[i])
        cluster_label = int(y_pred[i])
        contingency_matrix[cluster_label, true_label] += 1
    purity = np.sum(np.amax(contingency_matrix, axis=1)) / len(y_true)
    return purity

def visualize_centroids(centroids, save_path='centroids_plot.png'):
    """
    Visualize cluster centroids as images.
    Assumes centroids shape is (k, 784) where 784 = 28x28 (MNIST-like images)
    """
    k = centroids.shape[0]
    
    # Create a 2 row x 5 column grid for 10 clusters
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Cluster Centroids', fontsize=16)
    
    for i in range(k):
        row = i // 5
        col = i % 5
        
        # Reshape centroid from (784,) to (28, 28)
        centroid_img = centroids[i].reshape(28, 28)
        
        # Display as grayscale image
        axes[row, col].imshow(centroid_img, cmap='gray')
        axes[row, col].set_title(f'Cluster {i}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Centroid visualization saved to {save_path}")
    plt.close()

def main():
    data_path = 'kmeans_data/data.csv'
    label_path = 'kmeans_data/label.csv'
    X, y_true = load_data(data_path, label_path)
    
    k = 10
    metrics = ['euclidean', 'cosine', 'jaccard']
    
    results = {}
    
    print("\n--- Q1 & Q2: Comparing Metrics (Stop: OR condition) ---")
    best_centroids = None
    best_labels = None
    best_metric = 'cosine'  # Use cosine as it has best purity
    
    for m in metrics:
        print(f"Running {m}...")
        start_time = time.time()
        centroids, labels, iters, sse = kmeans(X, k, metric=m, stop_criteria='all')
        end_time = time.time()
        
        # Save centroids and labels for best metric (cosine)
        if m == best_metric:
            best_centroids = centroids
            best_labels = labels
            np.save('centroids.npy', centroids)
            np.save('labels_pred.npy', labels)
        
        purity = compute_purity(y_true, labels)
        results[m] = {
            'sse': sse,
            'purity': purity,
            'iters': iters,
            'time': end_time - start_time
        }
        print(f"  SSE: {sse:.4f}")
        print(f"  Purity: {purity:.4f}")
        print(f"  Iters: {iters}")
        print(f"  Time: {results[m]['time']:.4f}s")
    
    # Visualize cluster centroids (for cosine metric)
    print("\n--- Visualizing Cluster Centroids ---")
    visualize_centroids(best_centroids, save_path='centroids_plot.png')

    # Q4: Compare SSEs for specific terminating conditions
    print("\n--- Q4: Comparing Terminating Conditions ---")
    conditions = ['centroid_no_change', 'sse_increase', 'max_iters']
    # We need to run each metric with each condition?
    # The question says "Compare the SSEs of Euclidean... with respect to the following three terminating conditions".
    # So 3 metrics * 3 conditions.
    
    q4_results = {}
    for m in metrics:
        q4_results[m] = {}
        for c in conditions:
            print(f"Running {m} with {c}...")
            # For max_iters, let's set a fixed small number to see difference? Or use default 100?
            # The prompt says "preset value (e.g. 100)". Let's use 100.
            # For others, we use 500 as max to allow convergence.
            limit = 100 if c == 'max_iters' else 500
            
            centroids, labels, iters, sse = kmeans(X, k, metric=m, max_iters=limit, stop_criteria=c)
            q4_results[m][c] = sse
            print(f"  SSE: {sse:.4f}")

    # Save results to file for report
    with open('task1_results.txt', 'w') as f:
        f.write(str(results) + "\n")
        f.write(str(q4_results) + "\n")

if __name__ == "__main__":
    main()
