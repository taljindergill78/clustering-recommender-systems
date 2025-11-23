# Clustering and Recommender Systems

K-Means clustering implemented from scratch and collaborative filtering recommendation systems using Surprise library.

## 🎯 Overview

This project implements two major ML application areas:
1. **Clustering Analysis** - K-Means algorithm with multiple distance/similarity metrics
2. **Recommendation Systems** - Collaborative filtering and matrix factorization approaches

## 🔧 Implementations

### Task 1: K-Means Clustering
- **Algorithm**: K-Means clustering implemented from scratch
- **Distance Metrics**:
  - Euclidean distance
  - Cosine similarity (1 - cosine)
  - Generalized Jaccard similarity (1 - Jaccard)
- **Dataset**: 10,000 image samples with 784 features (28x28 pixels)
- **Analysis**:
  - SSE comparison across metrics
  - Clustering purity/accuracy evaluation
  - Convergence analysis (iterations and time)
  - Impact of different stopping criteria

**Key Findings**:
- Cosine similarity achieves 62.64% purity (best performance)
- Faster convergence with Cosine metric (28 iterations, 0.61s)
- High-dimensional data benefits from direction-based metrics over magnitude

### Task 2: Recommendation Systems
- **Implementation**: Using [Surprise Library](http://surpriselib.com/) (scikit-surprise)
- **Algorithms**:
  - Probabilistic Matrix Factorization (PMF) via SVD
  - User-based Collaborative Filtering with KNNWithMeans
  - Item-based Collaborative Filtering with KNNWithMeans
- **Similarity Metrics**: Cosine, MSD (Mean Squared Difference), Pearson
- **Dataset**: MovieLens Small (100,000+ ratings)
- **Evaluation**: 5-fold cross-validation with MAE and RMSE metrics

**Key Findings**:
- Results will be updated after running the code
- Multiple similarity metrics tested for comprehensive analysis
- K neighbors optimization from 5 to 60

## 📊 Results

### Clustering Performance
| Metric | SSE | Purity | Convergence Time |
|--------|-----|--------|------------------|
| Euclidean | 2.54e10 | 0.5851 | 4.28s |
| **Cosine** | **686.29** | **0.6264** | **0.61s** |
| Jaccard | 3,659.85 | 0.6012 | 5.84s |

### Recommendation System Performance (5-Fold CV)
**Note:** Results will be populated after running task2_recommender.py

| Algorithm | MAE | RMSE |
|-----------|-----|------|
| PMF | TBD | TBD |
| User-Based CF | TBD | TBD |
| Item-Based CF | TBD | TBD |

## 🚀 Usage

### K-Means Clustering
```python
python task1_kmeans.py
```
Generates:
- `centroids_plot.png` - Visualization of cluster centroids
- `centroids.npy` - Learned centroids
- `labels_pred.npy` - Cluster assignments
- `task1_results.txt` - Numerical results

### Recommendation Systems
```python
python task2_recommender.py
```
Generates:
- `similarity_impact.png` - Comparison of similarity metrics
- `k_impact.png` - Impact of number of neighbors
- `task2_results.txt` - Performance metrics

## 📦 Requirements

```
numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-surprise>=1.1.1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy matplotlib pandas scikit-surprise
```

## 📂 Project Structure

```
clustering-recommender-systems/
├── task1_kmeans.py              # K-Means implementation
├── task2_recommender.py         # Recommendation systems
├── kmeans_data/                 # Clustering dataset
│   ├── data.csv
│   └── label.csv
├── archive/                     # MovieLens dataset
│   └── ratings_small.csv
├── centroids_plot.png           # Output: cluster visualization
├── similarity_impact.png        # Output: similarity comparison
├── k_impact.png                 # Output: K neighbors analysis
└── README.md
```

## 📥 Dataset Information

### K-Means Dataset
Included in the repository under `kmeans_data/`:
- 10,000 samples with 784 features (28x28 pixel images)
- 10 class labels

### MovieLens Dataset
The `archive/` folder contains `ratings_small.csv` from the MovieLens dataset.

**Full Dataset (Optional):** If you want the complete MovieLens dataset, download it from:
- [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- Only `ratings_small.csv` (100k ratings) is needed for this implementation

## 🎓 Methodology

### Clustering
- Random centroid initialization (seed=42 for reproducibility)
- Multiple stopping criteria: centroid convergence, SSE increase, max iterations
- Majority vote labeling for purity calculation

### Recommendation Systems
- Implemented using Surprise library for optimized performance
- 5-fold cross-validation for robust evaluation
- K-nearest neighbors: tested range [5, 10, 15, ..., 60]
- PMF via SVD (20 factors, 20 epochs, unbiased)

## 📈 Visualizations

The project includes comprehensive visualizations:
- Cluster centroids as 28x28 grayscale images
- Similarity metric impact comparison (bar plots)
- K neighbors impact analysis (line plots)

## 🔍 Key Insights

1. **Metric Selection Matters**: For high-dimensional data (images), angle-based metrics (Cosine) outperform magnitude-based (Euclidean)

2. **Library vs From-Scratch**: Using optimized libraries (Surprise) provides better performance and reliability for production systems

3. **KNNWithMeans Advantage**: Using mean-adjusted collaborative filtering accounts for user/item rating biases, improving accuracy

4. **Comprehensive Evaluation**: Testing multiple K values and similarity metrics ensures optimal hyperparameter selection

## 🛠️ Implementation Details

- **Task 1 (K-Means)**: Implemented from scratch using only NumPy with vectorized operations
- **Task 2 (Recommender)**: Using Surprise library for professional-grade implementations
- **Multiple Metrics**: Comprehensive comparison across different approaches
- **Reproducible**: Fixed random seeds and documented parameters
- **Efficient**: Leverages optimized algorithms and data structures

## 📝 License

MIT License - feel free to use for learning and reference

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome!

---

**Built with** 🧠 **and** ☕

