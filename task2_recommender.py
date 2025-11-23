"""
Task 2: Recommender Systems using Surprise Library
Implements PMF, User-based CF, and Item-based CF using scikit-surprise
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNWithMeans
from surprise.model_selection import cross_validate
import time

def load_and_prepare_data(file_path='archive/ratings_small.csv'):
    """
    Load ratings data and prepare for Surprise library.
    Reads: userID, movieID, rating, timestamp
    """
    print(f"Loading ratings from {file_path}...")
    
    # Read CSV to verify format
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} ratings")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Users: {df['userId'].nunique()}, Movies: {df['movieId'].nunique()}")
    
    # Prepare data for Surprise (rating scale 0.5 to 5.0)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    return data, df

def evaluate_algorithms(data):
    """
    Q(c): Evaluate PMF, User-CF, and Item-CF using 5-fold cross-validation
    """
    print("\n" + "="*70)
    print("Q(c): 5-Fold Cross-Validation Results")
    print("="*70)
    
    # Define algorithms using Keval's approach (KNNWithMeans for best results)
    algorithms = {
        'PMF': SVD(n_factors=20, n_epochs=20, biased=False, random_state=42),
        'User-CF': KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': True}),
        'Item-CF': KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': False})
    }
    
    results = {}
    fold_details = {}
    
    for name, algo in algorithms.items():
        print(f"\nEvaluating {name}...")
        start_time = time.time()
        
        # Run 5-fold cross-validation
        cv_results = cross_validate(
            algo, data, 
            measures=['RMSE', 'MAE'], 
            cv=5, 
            verbose=True,
            return_train_measures=False
        )
        
        elapsed = time.time() - start_time
        
        # Store average results
        results[name] = {
            'mae': cv_results['test_mae'].mean(),
            'rmse': cv_results['test_rmse'].mean(),
            'mae_std': cv_results['test_mae'].std(),
            'rmse_std': cv_results['test_rmse'].std()
        }
        
        # Store individual fold details
        fold_details[name] = {
            'fold_maes': list(cv_results['test_mae']),
            'fold_rmses': list(cv_results['test_rmse']),
            'fold_fit_times': list(cv_results['fit_time']),
            'fold_test_times': list(cv_results['test_time'])
        }
        
        print(f"  Average MAE:  {results[name]['mae']:.4f} (±{results[name]['mae_std']:.4f})")
        print(f"  Average RMSE: {results[name]['rmse']:.4f} (±{results[name]['rmse_std']:.4f})")
        print(f"  Total time: {elapsed:.2f}s")
    
    return results, fold_details

def test_similarity_metrics(data):
    """
    Q(e): Test different similarity metrics for CF algorithms
    """
    print("\n" + "="*70)
    print("Q(e): Impact of Similarity Metrics")
    print("="*70)
    
    similarity_metrics = ['cosine', 'msd', 'pearson']
    sim_results = {'User-CF': {}, 'Item-CF': {}}
    sim_fold_details = {'User-CF': {}, 'Item-CF': {}}
    
    for sim in similarity_metrics:
        print(f"\nTesting {sim.upper()} similarity...")
        
        # User-based CF
        print(f"  User-based CF with {sim}...")
        user_algo = KNNWithMeans(k=40, sim_options={'name': sim, 'user_based': True})
        user_cv = cross_validate(user_algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        
        sim_results['User-CF'][sim] = {
            'mae': user_cv['test_mae'].mean(),
            'rmse': user_cv['test_rmse'].mean()
        }
        sim_fold_details['User-CF'][sim] = {
            'fold_maes': list(user_cv['test_mae']),
            'fold_rmses': list(user_cv['test_rmse']),
            'fold_fit_times': list(user_cv['fit_time']),
            'fold_test_times': list(user_cv['test_time'])
        }
        
        # Item-based CF
        print(f"  Item-based CF with {sim}...")
        item_algo = KNNWithMeans(k=40, sim_options={'name': sim, 'user_based': False})
        item_cv = cross_validate(item_algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        
        sim_results['Item-CF'][sim] = {
            'mae': item_cv['test_mae'].mean(),
            'rmse': item_cv['test_rmse'].mean()
        }
        sim_fold_details['Item-CF'][sim] = {
            'fold_maes': list(item_cv['test_mae']),
            'fold_rmses': list(item_cv['test_rmse']),
            'fold_fit_times': list(item_cv['fit_time']),
            'fold_test_times': list(item_cv['test_time'])
        }
        
        print(f"    User-CF: MAE={sim_results['User-CF'][sim]['mae']:.4f}, "
              f"RMSE={sim_results['User-CF'][sim]['rmse']:.4f}")
        print(f"    Item-CF: MAE={sim_results['Item-CF'][sim]['mae']:.4f}, "
              f"RMSE={sim_results['Item-CF'][sim]['rmse']:.4f}")
    
    return sim_results, sim_fold_details

def test_neighbor_impact(data):
    """
    Q(f) & Q(g): Test impact of number of neighbors (K)
    """
    print("\n" + "="*70)
    print("Q(f) & Q(g): Impact of Number of Neighbors (K)")
    print("="*70)
    
    k_values = list(range(5, 65, 5))  # [5, 10, 15, ..., 60]
    k_results = {'User-CF': [], 'Item-CF': []}
    
    for k in k_values:
        print(f"Testing K={k}...")
        
        # User-based CF
        user_algo = KNNWithMeans(k=k, sim_options={'name': 'cosine', 'user_based': True})
        user_cv = cross_validate(user_algo, data, measures=['RMSE'], cv=5, verbose=False)
        k_results['User-CF'].append(user_cv['test_rmse'].mean())
        
        # Item-based CF
        item_algo = KNNWithMeans(k=k, sim_options={'name': 'cosine', 'user_based': False})
        item_cv = cross_validate(item_algo, data, measures=['RMSE'], cv=5, verbose=False)
        k_results['Item-CF'].append(item_cv['test_rmse'].mean())
        
        print(f"  User-CF RMSE: {k_results['User-CF'][-1]:.4f}")
        print(f"  Item-CF RMSE: {k_results['Item-CF'][-1]:.4f}")
    
    # Find best K for each method
    best_k_user = k_values[np.argmin(k_results['User-CF'])]
    best_k_item = k_values[np.argmin(k_results['Item-CF'])]
    best_rmse_user = min(k_results['User-CF'])
    best_rmse_item = min(k_results['Item-CF'])
    
    print(f"\nBest K for User-CF: {best_k_user} (RMSE={best_rmse_user:.4f})")
    print(f"Best K for Item-CF: {best_k_item} (RMSE={best_rmse_item:.4f})")
    
    return k_values, k_results, best_k_user, best_k_item

def create_visualizations(sim_results, k_values, k_results):
    """
    Create visualization plots for similarity impact and K impact
    """
    print("\nGenerating visualization plots...")
    
    # Plot 1: Similarity Impact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = ['cosine', 'msd', 'pearson']
    x = np.arange(len(metrics))
    width = 0.35
    
    user_rmses = [sim_results['User-CF'][m]['rmse'] for m in metrics]
    item_rmses = [sim_results['Item-CF'][m]['rmse'] for m in metrics]
    
    ax1.bar(x - width/2, user_rmses, width, label='User-CF', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, item_rmses, width, label='Item-CF', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Similarity Metric')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Impact of Similarity Metrics on RMSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Cosine', 'MSD', 'Pearson'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    user_maes = [sim_results['User-CF'][m]['mae'] for m in metrics]
    item_maes = [sim_results['Item-CF'][m]['mae'] for m in metrics]
    
    ax2.bar(x - width/2, user_maes, width, label='User-CF', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, item_maes, width, label='Item-CF', color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Similarity Metric')
    ax2.set_ylabel('MAE')
    ax2.set_title('Impact of Similarity Metrics on MAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Cosine', 'MSD', 'Pearson'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('similarity_impact.png', dpi=150, bbox_inches='tight')
    print("  Saved: similarity_impact.png")
    plt.close()
    
    # Plot 2: K Impact
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, k_results['User-CF'], 'o-', label='User-CF', linewidth=2, markersize=6, color='#3498db')
    plt.plot(k_values, k_results['Item-CF'], 's-', label='Item-CF', linewidth=2, markersize=6, color='#e74c3c')
    plt.xlabel('Number of Neighbors (K)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Impact of Number of Neighbors on RMSE', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('k_impact.png', dpi=150, bbox_inches='tight')
    print("  Saved: k_impact.png")
    plt.close()

def save_results(results, fold_details, sim_results, sim_fold_details, 
                k_values, k_results, best_k_user, best_k_item):
    """
    Save all results to task2_results.txt for report generation
    """
    with open('task2_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("TASK 2 RESULTS - RECOMMENDER SYSTEMS (Surprise Library)\n")
        f.write("="*70 + "\n\n")
        
        # Q(c) Results
        f.write("Q(c): 5-Fold Cross-Validation Average Results\n")
        f.write("-"*70 + "\n")
        for name, res in results.items():
            f.write(f"{name}:\n")
            f.write(f"  MAE:  {res['mae']:.4f} (±{res['mae_std']:.4f})\n")
            f.write(f"  RMSE: {res['rmse']:.4f} (±{res['rmse_std']:.4f})\n\n")
        
        # Q(c) Individual Fold Details
        f.write("\nQ(c): Individual Fold Details\n")
        f.write("-"*70 + "\n")
        for name in ['PMF', 'User-CF', 'Item-CF']:
            f.write(f"\n{name} - Individual Folds:\n")
            folds = fold_details[name]
            for i in range(5):
                f.write(f"  Fold {i+1}: MAE={folds['fold_maes'][i]:.4f}, "
                       f"RMSE={folds['fold_rmses'][i]:.4f}, "
                       f"Fit={folds['fold_fit_times'][i]:.2f}s, "
                       f"Test={folds['fold_test_times'][i]:.2f}s\n")
        
        # Q(d) Best Model
        f.write("\n" + "="*70 + "\n")
        f.write("Q(d): Best Model Comparison\n")
        f.write("-"*70 + "\n")
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        f.write(f"Best Model (by RMSE): {best_model}\n")
        f.write(f"  MAE:  {results[best_model]['mae']:.4f}\n")
        f.write(f"  RMSE: {results[best_model]['rmse']:.4f}\n\n")
        
        # Q(e) Similarity Results
        f.write("="*70 + "\n")
        f.write("Q(e): Similarity Metrics Impact\n")
        f.write("-"*70 + "\n")
        for method in ['User-CF', 'Item-CF']:
            f.write(f"\n{method}:\n")
            for sim in ['cosine', 'msd', 'pearson']:
                f.write(f"  {sim.upper():8s}: MAE={sim_results[method][sim]['mae']:.4f}, "
                       f"RMSE={sim_results[method][sim]['rmse']:.4f}\n")
        
        # Q(e) Individual Fold Details for Similarity Metrics
        f.write("\n\nQ(e): Similarity Metrics - Individual Fold Details\n")
        f.write("-"*70 + "\n")
        for method in ['User-CF', 'Item-CF']:
            for sim in ['cosine', 'msd', 'pearson']:
                f.write(f"\n{method} - {sim.upper()}:\n")
                folds = sim_fold_details[method][sim]
                for i in range(5):
                    f.write(f"  Fold {i+1}: MAE={folds['fold_maes'][i]:.4f}, "
                           f"RMSE={folds['fold_rmses'][i]:.4f}\n")
        
        # Q(f) K Impact
        f.write("\n" + "="*70 + "\n")
        f.write("Q(f): Impact of Number of Neighbors (K)\n")
        f.write("-"*70 + "\n")
        f.write("K Values: " + str(k_values) + "\n\n")
        f.write("User-CF RMSE: " + str([f"{x:.4f}" for x in k_results['User-CF']]) + "\n")
        f.write("Item-CF RMSE: " + str([f"{x:.4f}" for x in k_results['Item-CF']]) + "\n\n")
        
        # Q(g) Best K
        f.write("="*70 + "\n")
        f.write("Q(g): Best K Values\n")
        f.write("-"*70 + "\n")
        f.write(f"Best K for User-CF: {best_k_user} (RMSE={min(k_results['User-CF']):.4f})\n")
        f.write(f"Best K for Item-CF: {best_k_item} (RMSE={min(k_results['Item-CF']):.4f})\n")
    
    print("\nResults saved to: task2_results.txt")

def main():
    """
    Main execution function
    """
    print("="*70)
    print("TASK 2: RECOMMENDER SYSTEMS")
    print("Implementation: Surprise Library (scikit-surprise)")
    print("="*70)
    
    # Load data
    data, df = load_and_prepare_data()
    
    # Q(c): Evaluate algorithms
    results, fold_details = evaluate_algorithms(data)
    
    # Q(e): Test similarity metrics
    sim_results, sim_fold_details = test_similarity_metrics(data)
    
    # Q(f) & Q(g): Test K values
    k_values, k_results, best_k_user, best_k_item = test_neighbor_impact(data)
    
    # Create visualizations
    create_visualizations(sim_results, k_values, k_results)
    
    # Save all results
    save_results(results, fold_details, sim_results, sim_fold_details,
                k_values, k_results, best_k_user, best_k_item)
    
    print("\n" + "="*70)
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - task2_results.txt")
    print("  - similarity_impact.png")
    print("  - k_impact.png")

if __name__ == "__main__":
    main()
