import csv
import math
import random
import numpy as np
import time
from collections import defaultdict

# --- Data Loading ---
def load_ratings(file_path):
    """
    Read data from "ratings small.csv" with line format: 'userID movieID rating timestamp'.
    
    """
    print(f"Loading ratings from {file_path}...")
    ratings = []
    users = set()
    items = set()
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            u = int(row[0])       # userID
            i = int(row[1])       # movieID
            r = float(row[2])     # rating
            t = int(row[3])       # timestamp
            ratings.append((u, i, r))  # Store (userID, movieID, rating) for algorithms
            users.add(u)
            items.add(i)
    print(f"Loaded {len(ratings)} ratings. Users: {len(users)}, Items: {len(items)}")
    return ratings, list(users), list(items)

# --- Similarity Metrics ---
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0
    return dot / (norm1 * norm2)

def msd_similarity(v1, v2):
    # MSD = sum((a-b)^2) / |common|
    # Sim = 1 / (MSD + 1)
    # Here v1, v2 are vectors of ratings on COMMON items only
    diff = v1 - v2
    msd = np.mean(diff**2)
    return 1 / (msd + 1)

def pearson_similarity(v1, v2):
    # Center data
    v1_c = v1 - np.mean(v1)
    v2_c = v2 - np.mean(v2)
    return cosine_similarity(v1_c, v2_c)

# --- Models ---

class CollaborativeFiltering:
    def __init__(self, mode='user', metric='cosine', k=20):
        self.mode = mode # 'user' or 'item'
        self.metric = metric
        self.k = k
        self.user_items = defaultdict(dict)
        self.item_users = defaultdict(dict)
        self.means = {}
        self.similarities = {}
        
    def fit(self, train_data):
        self.user_items = defaultdict(dict)
        self.item_users = defaultdict(dict)
        all_ratings = defaultdict(list)
        
        for u, i, r in train_data:
            self.user_items[u][i] = r
            self.item_users[i][u] = r
            if self.mode == 'user':
                all_ratings[u].append(r)
            else:
                all_ratings[i].append(r)
                
        # Compute means
        self.means = {k: np.mean(v) for k, v in all_ratings.items()}
        
        # Precompute similarities? 
        # For 100k ratings, computing all-pairs is slow.
        # We will compute on demand or use a simplified approach for this HW.
        # Given the constraints and "from scratch", let's implement a lazy or memory-based approach.
        # To make it feasible, we'll compute similarities only for relevant neighbors during prediction
        # OR precompute a subset.
        # Actually, for User-Based (671 users), we CAN precompute 671x671 matrix.
        # For Item-Based (9000 items), 9000x9000 is 81M entries. Too big/slow.
        # So for Item-Based, we might need to be smarter or just slow.
        # Let's precompute for User, and lazy for Item? Or just lazy for both.
        
        if self.mode == 'user':
            self.compute_user_similarities()
        
    def compute_user_similarities(self):
        users = list(self.user_items.keys())
        n = len(users)
        self.similarities = {}
        
        # Convert to dense vectors for common items is tricky.
        # Let's just store the user_items dicts and compute pairwise.
        
        # Optimization: Pre-calculate norms/means if needed.
        
        pass # We will do it in predict for simplicity/flexibility with metrics
        
    def get_similarity(self, id1, id2, data_dict):
        # data_dict is user_items if mode='user' (comparing users)
        # data_dict is item_users if mode='item' (comparing items)
        
        items1 = data_dict[id1]
        items2 = data_dict[id2]
        
        common_keys = set(items1.keys()) & set(items2.keys())
        if not common_keys:
            return 0
            
        v1 = np.array([items1[k] for k in common_keys])
        v2 = np.array([items2[k] for k in common_keys])
        
        if self.metric == 'cosine':
            return cosine_similarity(v1, v2)
        elif self.metric == 'msd':
            return msd_similarity(v1, v2)
        elif self.metric == 'pearson':
            return pearson_similarity(v1, v2)
        return 0

    def predict(self, u, i):
        # User-based: predict r_ui based on neighbors of u who rated i
        # Item-based: predict r_ui based on neighbors of i rated by u
        
        if self.mode == 'user':
            if u not in self.user_items: return 3.0
            target_id = u
            candidate_ids = self.item_users.get(i, {}).keys() # Users who rated i
            data_dict = self.user_items
        else:
            if i not in self.item_users: return 3.0
            target_id = i
            candidate_ids = self.user_items.get(u, {}).keys() # Items rated by u
            data_dict = self.item_users
            
        candidates = []
        for cand_id in candidate_ids:
            if cand_id == target_id: continue
            sim = self.get_similarity(target_id, cand_id, data_dict)
            if sim > 0:
                candidates.append((sim, cand_id))
                
        # Top K
        candidates.sort(key=lambda x: x[0], reverse=True)
        k_neighbors = candidates[:self.k]
        
        if not k_neighbors:
            return self.means.get(target_id, 3.0)
            
        numerator = 0
        denominator = 0
        
        for sim, cand_id in k_neighbors:
            # Rating of candidate on the target item/user
            # If User-based: cand_id is a user, we need their rating on item i
            # If Item-based: cand_id is an item, we need user u's rating on it
            if self.mode == 'user':
                r = self.user_items[cand_id][i]
            else:
                r = self.item_users[cand_id][u]
                
            numerator += sim * r
            denominator += sim
            
        if denominator == 0:
            return self.means.get(target_id, 3.0)
            
        return numerator / denominator

class PMF:
    def __init__(self, n_factors=10, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_factors = {}
        self.item_factors = {}
        
    def fit(self, train_data):
        users = set(x[0] for x in train_data)
        items = set(x[1] for x in train_data)
        
        # Initialize factors
        np.random.seed(42)
        self.user_factors = {u: np.random.normal(0, 0.1, self.n_factors) for u in users}
        self.item_factors = {i: np.random.normal(0, 0.1, self.n_factors) for i in items}
        
        for epoch in range(self.n_epochs):
            # SGD
            # Shuffle data?
            # np.random.shuffle(train_data) # List shuffle is in-place
            
            err_sum = 0
            for u, i, r in train_data:
                if u not in self.user_factors or i not in self.item_factors: continue
                
                pred = np.dot(self.user_factors[u], self.item_factors[i])
                err = r - pred
                err_sum += err**2
                
                # Update
                u_f = self.user_factors[u]
                i_f = self.item_factors[i]
                
                self.user_factors[u] += self.lr * (err * i_f - self.reg * u_f)
                self.item_factors[i] += self.lr * (err * u_f - self.reg * i_f)
                
            # print(f"PMF Epoch {epoch+1}: MSE = {err_sum/len(train_data):.4f}")

    def predict(self, u, i):
        if u in self.user_factors and i in self.item_factors:
            val = np.dot(self.user_factors[u], self.item_factors[i])
            return min(5, max(0.5, val)) # Clip
        return 3.0 # Default

# --- Evaluation ---
def cross_validate(model_class, data, k_folds=5, **model_kwargs):
    # Split data
    random.seed(42)
    random.shuffle(data)
    fold_size = len(data) // k_folds
    
    maes = []
    rmses = []
    fold_details = []  # NEW: Store individual fold results
    
    for k in range(k_folds):
        start = k * fold_size
        end = (k + 1) * fold_size
        test_fold = data[start:end]
        train_fold = data[:start] + data[end:]
        
        fit_start = time.time()  # NEW: Track fit time
        model = model_class(**model_kwargs)
        model.fit(train_fold)
        fit_time = time.time() - fit_start  # NEW
        
        test_start = time.time()  # NEW: Track test time
        mae_sum = 0
        rmse_sum = 0
        n = 0
        
        for u, i, r in test_fold:
            pred = model.predict(u, i)
            err = r - pred
            mae_sum += abs(err)
            rmse_sum += err**2
            n += 1
            
        mae = mae_sum / n
        rmse = math.sqrt(rmse_sum / n)
        test_time = time.time() - test_start  # NEW
        
        maes.append(mae)
        rmses.append(rmse)
        fold_details.append({  # NEW: Store fold details
            'fold': k+1,
            'mae': mae,
            'rmse': rmse,
            'fit_time': fit_time,
            'test_time': test_time
        })
        print(f"Fold {k+1}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
    return np.mean(maes), np.mean(rmses), fold_details  # NEW: Return fold details

def main():
    ratings, _, _ = load_ratings('archive/ratings_small.csv')
    # Use a subset for speed if needed? 100k is okay for PMF and User-CF. Item-CF might be slow.
    # Let's use full dataset.
    
    results = {}
    fold_results = {}  # NEW: Store individual fold results
    
    # Q(c): Average MAE/RMSE for PMF, User-CF, Item-CF
    print("\n--- Q(c): 5-Fold CV ---")
    
    print("Running PMF...")
    pmf_mae, pmf_rmse, pmf_folds = cross_validate(PMF, ratings, n_factors=20, n_epochs=10)
    results['PMF'] = (pmf_mae, pmf_rmse)
    fold_results['PMF'] = pmf_folds  # NEW
    print(f"PMF Average: MAE={pmf_mae:.4f}, RMSE={pmf_rmse:.4f}")
    
    print("Running User-Based CF (Cosine)...")
    ub_mae, ub_rmse, ub_folds = cross_validate(CollaborativeFiltering, ratings, mode='user', metric='cosine')
    results['UserCF'] = (ub_mae, ub_rmse)
    fold_results['UserCF'] = ub_folds  # NEW
    print(f"UserCF Average: MAE={ub_mae:.4f}, RMSE={ub_rmse:.4f}")
    
    print("Running Item-Based CF (Cosine)...")
    # Item-based is slow. Let's do 2 folds or just 1 for demonstration if it takes too long?
    # The requirement is 5-fold. I'll try to optimize or just wait.
    # To speed up, I'll reduce K or something? No, K is for prediction.
    # I'll just run it.
    ib_mae, ib_rmse, ib_folds = cross_validate(CollaborativeFiltering, ratings, mode='item', metric='cosine')
    results['ItemCF'] = (ib_mae, ib_rmse)
    fold_results['ItemCF'] = ib_folds  # NEW
    print(f"ItemCF Average: MAE={ib_mae:.4f}, RMSE={ib_rmse:.4f}")
    
    # Q(e): Impact of Similarity Metrics
    print("\n--- Q(e): Impact of Similarity Metrics ---")
    metrics = ['cosine', 'msd', 'pearson']
    sim_results = {'user': {}, 'item': {}}
    sim_fold_results = {'user': {}, 'item': {}}  # NEW: Store fold details
    
    # Run 5-fold CV for each metric to match friend's detail level
    for m in metrics:
        print(f"\nTesting User-CF with {m} (5-fold CV)...")
        user_mae, user_rmse, user_folds = cross_validate(CollaborativeFiltering, ratings, mode='user', metric=m)
        sim_results['user'][m] = (user_mae, user_rmse)
        sim_fold_results['user'][m] = user_folds  # NEW
        print(f"User-CF {m}: MAE={user_mae:.4f}, RMSE={user_rmse:.4f}")
        
        print(f"Testing Item-CF with {m} (5-fold CV)...")
        item_mae, item_rmse, item_folds = cross_validate(CollaborativeFiltering, ratings, mode='item', metric=m)
        sim_results['item'][m] = (item_mae, item_rmse)
        sim_fold_results['item'][m] = item_folds  # NEW
        print(f"Item-CF {m}: MAE={item_mae:.4f}, RMSE={item_rmse:.4f}")

    # Q(f): Impact of Neighbors (K)
    print("\n--- Q(f): Impact of Neighbors (K) ---")
    k_values = list(range(5, 65, 5))  # [5,10,15,20,25,30,35,40,45,50,55,60] - More granular
    k_results = {'user': [], 'item': []}
    
    # Use 80/20 split for K testing (faster than 5-fold)
    train_size = int(0.8 * len(ratings))
    train_data = ratings[:train_size]
    test_data = ratings[train_size:]
    
    for k in k_values:
        print(f"Testing K={k}...")
        # User CF
        model = CollaborativeFiltering(mode='user', metric='cosine', k=k)
        model.fit(train_data)
        rmse_sum, n = 0, 0
        for u, i, r in test_data:
            pred = model.predict(u, i)
            rmse_sum += (r - pred)**2
            n += 1
        k_results['user'].append(math.sqrt(rmse_sum/n))
        
        # Item CF
        model = CollaborativeFiltering(mode='item', metric='cosine', k=k)
        model.fit(train_data)
        rmse_sum, n = 0, 0
        for u, i, r in test_data:
            pred = model.predict(u, i)
            rmse_sum += (r - pred)**2
            n += 1
        k_results['item'].append(math.sqrt(rmse_sum/n))

    # Save results
    with open('task2_results.txt', 'w') as f:
        f.write("=== MAIN RESULTS ===\n")
        f.write(str(results) + "\n\n")
        f.write("=== FOLD DETAILS (c) ===\n")
        f.write(str(fold_results) + "\n\n")
        f.write("=== SIMILARITY RESULTS ===\n")
        f.write(str(sim_results) + "\n\n")
        f.write("=== SIMILARITY FOLD DETAILS (e) ===\n")
        f.write(str(sim_fold_results) + "\n\n")
        f.write("=== K VALUES ===\n")
        f.write(str(k_values) + "\n\n")
        f.write("=== K RESULTS ===\n")
        f.write(str(k_results) + "\n")
        
    # Plotting
    import matplotlib.pyplot as plt
    
    # Plot Similarity Impact
    labels = metrics
    user_rmses = [sim_results['user'][m][1] for m in metrics]
    item_rmses = [sim_results['item'][m][1] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, user_rmses, width, label='User CF')
    rects2 = ax.bar(x + width/2, item_rmses, width, label='Item CF')
    
    ax.set_ylabel('RMSE')
    ax.set_title('Impact of Similarity Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('similarity_impact.png')
    
    # Plot K Impact
    plt.figure()
    plt.plot(k_values, k_results['user'], label='User CF')
    plt.plot(k_values, k_results['item'], label='Item CF')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('RMSE')
    plt.title('Impact of Neighbors (K)')
    plt.legend()
    plt.savefig('k_impact.png')

if __name__ == "__main__":
    main()
