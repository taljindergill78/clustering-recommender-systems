# Task 2 Implementation Changes Summary

## 🔄 What Changed?

### **FROM:** Custom From-Scratch Implementation
- Custom PMF with SGD
- Custom User/Item-based CF
- ~410 lines of code
- Results: 2-10% worse than optimized libraries

### **TO:** Surprise Library Implementation (Keval's Approach)
- Uses `SVD` for PMF (unbiased, 20 factors, 20 epochs)
- Uses `KNNWithMeans` for User/Item-CF (K=40, cosine similarity)
- ~250 lines of clean code
- **Expected Results:** Best performance among all implementations

---

## 📂 Files Modified

### 1. **task2_recommender.py** - COMPLETE REWRITE ✅
**New Implementation:**
- Uses Surprise library (`SVD`, `KNNWithMeans`)
- Follows professor's RECOMMENDED approach (Surprise is listed first in assignment)
- Implements all required analyses:
  - Q(c): 5-fold CV for PMF, User-CF, Item-CF
  - Q(d): Best model comparison
  - Q(e): Similarity metrics impact (cosine, msd, pearson)
  - Q(f): K neighbors impact (K=5 to 60)
  - Q(g): Best K identification
- Saves individual fold details for comprehensive reporting
- Generates same output files: `task2_results.txt`, `similarity_impact.png`, `k_impact.png`

### 2. **requirements.txt** - UPDATED ✅
**Added:**
```
pandas>=1.3.0
scikit-surprise>=1.1.1
```

### 3. **README.md** - UPDATED ✅
**Changes:**
- Updated description to mention Surprise library
- Updated Task 2 implementation details
- Updated requirements section
- Marked results as "TBD" until you run the new code
- Updated implementation details and key insights

### 4. **report.md & report.html** - NEED UPDATES (After you run code)
**Changes Needed:**
- Update methodology to mention Surprise library
- Replace results with new values from `task2_results.txt`
- Keep all analysis sections (just update numbers)
- Add note about using KNNWithMeans (accounts for rating biases)

---

## 🚀 Next Steps

### Step 1: Install Dependencies
```bash
cd "/Users/taljindersingh/Documents/Personal Space/ARIZONA STATE UNIVERSITY/Academics/3rd Semester (Fall 2025)/CSE 572 - Data Mining/HW/HW3"

pip install scikit-surprise pandas
```

### Step 2: Run Task 2
```bash
python task2_recommender.py
```

**This will generate:**
- `task2_results.txt` - All numerical results for the report
- `similarity_impact.png` - Updated visualization
- `k_impact.png` - Updated visualization

### Step 3: Share Results with Me
Once the code finishes running:
1. Share the `task2_results.txt` file content
2. Share the new plot images if needed
3. I'll update the report with the new results

### Step 4: Compare with Friends
After getting results, we'll compare:
- Your new results vs Mohan's Surprise implementation
- Your new results vs Keval's KNNWithMeans implementation
- Verify you're getting similar or better performance

---

## 📊 Expected Results (Based on Keval's Implementation)

### Q(c): Algorithm Performance
- **PMF**: MAE ~0.74-0.75, RMSE ~0.97-0.98
- **User-CF**: MAE ~0.70-0.71, RMSE ~0.92-0.93 🏆 (Best)
- **Item-CF**: MAE ~0.70-0.71, RMSE ~0.92-0.93

### Q(d): Best Model
- Expected: **User-based CF** (lowest errors)

### Q(e): Similarity Metrics
- All three metrics (cosine, msd, pearson) will be tested
- Results should show consistent patterns
- Individual fold details will be included

### Q(f) & Q(g): K Neighbors
- K values: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
- Expected best K for User-CF: ~40-50
- Expected best K for Item-CF: ~40-50
- Plots will show smooth convergence curves

---

## ✅ Why This Change is Good

### 1. **Follows Professor's Guidelines** ✅
- Surprise library is the **FIRST option** listed in the assignment
- Using recommended tools shows good judgment
- Still demonstrates understanding through proper usage

### 2. **Better Results** ✅
- Expected 5-10% improvement in MAE/RMSE
- Uses KNNWithMeans (accounts for user/item biases)
- Professionally optimized implementations

### 3. **Cleaner Code** ✅
- ~250 lines vs ~410 lines
- More maintainable
- Industry-standard approach

### 4. **No Impact on Task 1** ✅
- Task 1 remains unchanged
- Your excellent from-scratch K-Means implementation stays
- Best of both worlds: demonstrate scratch implementation AND library usage

---

## 🎯 Final Comparison After Implementation

Once you run the code, we'll compare:

| Aspect | YOUR OLD (Scratch) | YOUR NEW (Surprise) | MOHAN | KEVAL |
|--------|-------------------|---------------------|-------|-------|
| **Approach** | From Scratch | Surprise (KNNWithMeans) | Surprise (KNNBasic) | Surprise (KNNWithMeans) |
| **Expected PMF** | 0.837/1.105 | **~0.745/0.974** | 0.777/1.006 | **0.746/0.974** |
| **Expected User-CF** | 0.762/0.989 | **~0.707/0.923** | 0.745/0.968 | **0.707/0.923** |
| **Expected Item-CF** | 0.792/1.014 | **~0.709/0.927** | 0.722/0.935 | **0.709/0.927** |

Your new results should **match or beat** both Mohan and Keval!

---

## ❓ Questions?

**Q: Will this affect my Task 1 code?**
A: NO! Task 1 remains completely unchanged.

**Q: Is this allowed by the assignment?**
A: YES! It's the professor's FIRST recommended option.

**Q: Will I lose points for not implementing from scratch?**
A: NO! The assignment explicitly allows using Surprise library.

**Q: What if my results differ slightly from Keval's?**
A: Small variations (0.1-0.5%) are normal due to random splits.

---

## 📝 After Running - Report Updates Needed

I'll help you update these sections in the report:

1. **Task 2 Introduction** - Mention using Surprise library
2. **Q(c) Results** - Update MAE/RMSE values and fold tables
3. **Q(d) Answer** - Update best model based on new results
4. **Q(e) Results** - Update similarity comparison tables and plots
5. **Q(f) Plot** - Replace K impact plot
6. **Q(g) Answer** - Update best K values

---

**Ready to run the new code and see the improved results! 🚀**

