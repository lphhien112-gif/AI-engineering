# 🎯 Classical ML — Câu Hỏi Phỏng Vấn (50+)

> Mỗi câu có **trả lời chi tiết**, follow-up questions, code khi cần, trade-offs.
> Strategy: giải thích concept → ví dụ → trade-off → khi nào dùng.

---

## Supervised Learning (15 câu)

### Q1: Bias-Variance tradeoff giải thích?
**A**: 
- **Bias** = error from wrong assumptions (underfitting). Ví dụ: dùng linear model cho non-linear data.
- **Variance** = error from sensitivity to training data (overfitting). Ví dụ: decision tree depth=unlimited → memorize noise.
- **Total Error = Bias² + Variance + Irreducible Noise**
- Low bias + High variance = overfit (complex model). High bias + Low variance = underfit (simple model).
- **Fix overfitting**: regularization, more data, simpler model, dropout, early stopping.
- **Fix underfitting**: more features, complex model, reduce regularization.
- **Follow-up**: "Training loss low, validation loss high → variance problem (overfit)."

### Q2: Random Forest vs XGBoost?
**A**: 
| | Random Forest | XGBoost |
|-|--------------|---------|
| Strategy | Bagging (parallel trees) | Boosting (sequential, learn errors) |
| Reduces | Variance | Bias |
| Tuning | Robust, ít hyper | Sensitive, nhiều params |
| Speed | Parallel (fast training) | Sequential (slower) |
| Accuracy | Good baseline | Usually better |
| Overfitting | Resistant | Needs regularization |

**Code**: `from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier`
**Follow-up**: "XGBoost thêm L1/L2 regularization + column subsampling → controls overfitting better than plain GBM."

### Q3: Linear Regression assumptions?
**A**: 5 assumptions:
1. **Linearity**: Y = β₀ + β₁X₁ + ... (linear relationship)
2. **Independence**: errors independent of each other
3. **Homoscedasticity**: constant error variance
4. **Normality**: residuals normally distributed
5. **No multicollinearity**: features not highly correlated

**Violations**: plot residuals to check. If violated → Ridge/Lasso (multicollinearity), polynomial features (non-linearity), weighted LS (heteroscedasticity).

### Q4: Logistic Regression có phải regression không?
**A**: **Không!** Classification algorithm.
- Output: `P(y=1|x) = sigmoid(w·x + b) = 1/(1 + exp(-z))`
- Loss: Binary Cross-Entropy = `-[y·log(p) + (1-y)·log(1-p)]`
- Decision boundary: linear (hyperplane in feature space)
- **Follow-up**: "Tại sao gọi 'Regression'?" → Historical name. Log-odds (logit) = log(p/(1-p)) = w·x + b IS actually a regression on log-odds.

### Q5: Ridge vs Lasso?
**A**:
| | Ridge (L2) | Lasso (L1) | ElasticNet |
|-|-----------|-----------|------------|
| Penalty | Σ w² | Σ \|w\| | α×L1 + (1-α)×L2 |
| Weights | Shrink → small | Some → **exactly 0** | Mix |
| Feature selection | ❌ No | ✅ Yes | ✅ Yes |
| Correlated features | Keeps all (spread weights) | Drops randomly | Handles well |

**Rule**: Many features, want sparse → Lasso. Multicollinearity → Ridge. Both → ElasticNet.

### Q6: SVM kernel trick?
**A**: Map data to higher dimension → linearly separable. Trick: compute dot product in high-dim space WITHOUT explicitly mapping.
- **Linear**: K(x,y) = x·y (fast, high-dim data like text)
- **RBF**: K(x,y) = exp(-γ||x-y||²) (default, infinite-dim, most flexible)
- **Polynomial**: K(x,y) = (x·y + c)^d
- **Follow-up**: "C parameter?" → regularization. Large C → small margin, fit training data harder. Small C → large margin, more generalization.

### Q7: Decision Tree overfitting?
**A**: Tree grows until pure leaves → memorize noise. Prevention:
- `max_depth` (most important)
- `min_samples_leaf` (require N samples per leaf)
- `min_samples_split`
- Post-pruning (cost-complexity pruning: `ccp_alpha`)
- **Best fix**: use ensemble (RF, XGBoost) instead of single tree

### Q8: OOB Score trong Random Forest?
**A**: Out-Of-Bag Score. Bootstrap sampling → each tree uses ~63.2% data. Remaining ~36.8% = "free" validation set.
- Each data point predicted by trees that DIDN'T train on it
- OOB score ≈ CV accuracy → no need for separate val set
- `RandomForestClassifier(oob_score=True)`

### Q9: Gradient Boosting cơ chế?
**A**: 
```
1. Fit tree₁ on data → predictions ŷ₁
2. Compute residuals: r₁ = y - ŷ₁
3. Fit tree₂ on RESIDUALS r₁ → δ₂
4. Update: ŷ₂ = ŷ₁ + η × δ₂ (η = learning rate)
5. Repeat: each tree corrects previous errors
```
- **Learning rate (η)**: smaller = better but needs more trees. Typical 0.01-0.1.
- **Follow-up**: "XGBoost improvements?" → L1/L2 regularization on leaf weights, column subsampling, parallel feature computation, tree pruning with gain.

### Q10: KNN curse of dimensionality?
**A**: High dimensions → all points equidistant → distance metric meaningless → KNN fails.
- Volume of hypersphere → 0 as d → ∞
- Need exponentially more data in high dims
- **Fix**: PCA reduce dim first, feature selection, use tree-based models, domain-specific distance metrics

### Q11: Multicollinearity ảnh hưởng gì?
**A**: Features highly correlated → coefficients **unstable** (huge, opposite signs), high standard errors.
- Model still predicts OK, but coefficients **uninterpretable**
- VIF (Variance Inflation Factor): VIF > 10 → problematic
- **Fix**: Drop correlated features, Ridge regression, PCA
- **Follow-up**: "Tree models affected?" → No, tree-based models handle collinearity naturally.

### Q12: Imbalanced data strategies?
**A**: 5 strategies (in order):
1. **Metrics first**: F1, AUC-PR (NOT accuracy)
2. **Class weights**: `class_weight="balanced"` (cheapest fix!)
3. **Threshold tuning**: PR curve → optimal threshold
4. **Resampling**: SMOTE (oversample minority), RandomUnderSampler
5. **Algorithmic**: Cost-sensitive learning, anomaly detection approach
- **Follow-up**: "SMOTE pitfalls?" → Generates synthetic samples in feature space → can create unrealistic samples. Apply AFTER train/test split only!

### Q13: Naive Bayes tại sao "Naive"?
**A**: Assumes features are **conditionally independent** given class: P(x₁,x₂|y) = P(x₁|y)·P(x₂|y). 
- Almost always wrong, but works surprisingly well for text (spam detection, sentiment)
- Very fast training + inference
- **Variants**: GaussianNB (continuous), MultinomialNB (text), BernoulliNB (binary)

### Q14: XGBoost hyperparameter tuning order?
**A**: 
1. `n_estimators` + `learning_rate` (start 0.1, 100-500 trees)
2. `max_depth` (3-8, most impactful)
3. `subsample` + `colsample_bytree` (0.6-0.9, regularization)
4. `min_child_weight` (prevent overfitting on rare events)
5. `reg_alpha` + `reg_lambda` (L1/L2 regularization)
6. Lower `learning_rate` to 0.01, increase `n_estimators` proportionally

### Q15: Time series → ML khác gì normal tabular?
**A**: 
- **Split**: time-based (NEVER random shuffle!)
- **CV**: `TimeSeriesSplit` (train on past, test on future)
- **Features**: lag features (t-1, t-2, ...), rolling stats, datetime features
- **Leakage danger**: future info leaking into past (e.g., future rolling mean)

---

## Feature Engineering (8 câu)

### Q16: Feature scaling khi nào cần?
**A**: 
| Cần | Không cần |
|-----|-----------|
| KNN, SVM (distance-based) | Decision Tree, RF, XGBoost |
| PCA (variance-based) | LightGBM |
| Neural Networks (gradient-based) | Rule-based systems |
| Logistic Regression | |

**Types**: StandardScaler (zero mean, unit var), MinMaxScaler (0-1), RobustScaler (median/IQR, handles outliers).

### Q17: One-Hot vs Target Encoding?
**A**:
- **One-Hot**: nominal, low cardinality (<20). Creates N-1 binary columns. Dense with many categories.
- **Target Encoding**: high cardinality (city, zip). Replace with mean of target. ⚠️ Needs smoothing + cross-fold to prevent overfitting.
- **Ordinal**: ordered categories (education level). Map to integers.
- **Follow-up**: "CatBoost handles categoricals natively — auto target encoding."

### Q18: Data leakage phòng tránh?
**A**: Split FIRST, then preprocess. Common leaks:
- Fit scaler on ALL data (including test)
- Target-encode on full dataset
- Using future data as features
- Duplicate rows in train AND test
- **Fix**: sklearn Pipeline → `fit` only on train, `transform` both

### Q19: Missing values handle thế nào?
**A**: 
- Numerical: `median` (robust to outliers) or `mean`
- Categorical: `most_frequent` or dedicated "Missing" category
- Advanced: `KNNImputer`, `IterativeImputer` (MICE)
- Always add `is_missing` binary indicator column
- **MAR vs MCAR vs MNAR**: if missingness is informative → MNAR → is_missing = valuable feature

### Q20: SHAP vs Feature Importance?
**A**:
| | Gini/Permutation Importance | SHAP |
|-|---------------------------|------|
| Scope | Global only | Global + **per-prediction** |
| Game theory | No | Yes (Shapley values) |
| Direction | Rankings only | Shows **positive/negative** effect |
| Interactions | Misses | SHAP interaction values |
| Speed | Fast | Slower |

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # Global
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])  # Local
```

### Q21: VIF (Variance Inflation Factor)?
**A**: `VIF_j = 1 / (1 - R²_j)` where R²_j = regressing feature j on all others.
- VIF = 1: no collinearity
- VIF = 5-10: moderate
- VIF > 10: problematic → drop or combine
- **Iterative process**: compute VIF → drop highest → recompute → repeat

### Q22: Feature creation strategies?
**A**: 
- **Interactions**: `price × quantity = revenue`, `height / weight = BMI`
- **Time**: day_of_week, is_weekend, hour_bucket, days_since_X
- **Aggregations**: user-level stats (mean purchase, count visits)
- **Text stats**: word_count, char_count, has_url, sentiment
- **Log/sqrt**: reduce skewness for right-skewed distributions
- **Cyclical**: `sin(2π × hour/24)`, `cos(2π × hour/24)` for circular features

### Q23: Feature Selection methods?
**A**: 
1. **Filter**: chi-squared, mutual info, correlation → fast, model-agnostic
2. **Wrapper**: RFE (recursive feature elimination) → uses model, expensive
3. **Embedded**: Lasso (L1 → zero weights), tree importance → during training
4. **Modern**: Boruta (RF-based), SHAP importance
- **Practical**: start with correlation filter → Lasso → SHAP for final selection

---

## Evaluation & Tuning (12 câu)

### Q24: Accuracy tại sao misleading?
**A**: 99:1 data → predict ALL negative → 99% accuracy, 0% recall, 0% F1. **Always check F1 + AUC for imbalanced**.

### Q25: Precision vs Recall — chọn theo business
**A**:
| Optimize | Khi | Ví dụ |
|----------|-----|-------|
| Precision | FP costly | Spam (mất email quan trọng), recommendation |
| Recall | FN costly | Cancer detection (miss cancer = life-threatening) |
| F1 | Balance | General |
| F2 | Recall ưu tiên hơn | Medical screening, fraud |

### Q26: AUC-ROC vs AUC-PR?
**A**: AUC-ROC can be misleadingly high with imbalanced data (high FPR doesn't matter when TN is huge). AUC-PR focuses on positive class → honest for imbalanced. **Rule: >10:1 imbalance → use AUC-PR.**

### Q27: Cross-validation types?
**A**:
| Type | When | Key |
|------|------|-----|
| StratifiedKFold | Classification (default) | Preserves class ratios |
| TimeSeriesSplit | Temporal data | Train past, test future |
| GroupKFold | Grouped data | Same group stays together |
| LeaveOneOut | Tiny datasets (<100) | K=N, expensive |
| RepeatedKFold | Reduce variance | K×R evaluations |

### Q28: Nested CV giải thích?
**A**: `Outer loop`: evaluate generalization. `Inner loop`: tune hyperparameters.
```
Outer fold 1: [Train + Inner CV for tuning] → eval on outer test
Outer fold 2: [Train + Inner CV for tuning] → eval on outer test
...
Result = mean of outer fold scores
```
Prevents overfitting during tuning. Mandatory for papers/unbiased estimates.

### Q29: Grid vs Random vs Bayesian?
**A**: Grid: exhaustive O(n^d). Random: 60 iterations ≈ 95% of grid's best. Bayesian (Optuna): learns from history → most efficient.
- Start → Random Search (fast exploration)
- Refine → Optuna (targeted optimization)
- Never → Grid Search (wastes compute on unimportant params)

### Q30: Early stopping khi nào?
**A**: Monitor validation metric, stop when no improvement for N epochs/rounds.
- XGBoost: `early_stopping_rounds=50`
- PyTorch: `EarlyStopping(patience=10)`
- Benefits: prevent overfitting + save time + find optimal iteration count

### Q31: sklearn Pipeline why?
**A**: 3 reasons:
1. **Prevent leakage**: fit preprocessing on train ONLY
2. **Reproducibility**: 1 object = preprocessor + model
3. **Deployment**: save/load 1 file, no manual preprocessing at serving

### Q32: Ensemble strategies?
**A**:
- **Bagging**: parallel, reduce variance (RF)
- **Boosting**: sequential, reduce bias (XGBoost, LightGBM)
- **Stacking**: train meta-model on base predictions (best but complex)
- **Blending**: weighted average (simple but effective)
- **Voting**: majority vote (classification)

### Q33: Model selection workflow?
**A**:
```
1. Baseline: LogisticRegression / DummyClassifier
2. Quick benchmark: RF, LightGBM, XGBoost (default params)
3. Pick top 2-3 → tune with Optuna
4. Final: Ensemble or best single model
5. Report: CV + test scores + confidence intervals
```

### Q34: Class weight vs SMOTE?
**A**: Class weight (`balanced`) → adjust loss function → no new data. SMOTE → generate synthetic minority samples → new data. 
- Start with class weight (zero effort)
- Try SMOTE if weight insufficient
- SMOTE only on TRAIN set, never test!

### Q35: Calibration — khi nào cần?
**A**: When you need reliable probabilities (medical, credit scoring). XGBoost → well-calibrated. NN/SVM → often poorly calibrated. 
- Fix: `CalibratedClassifierCV(method='isotonic')`
- Check: reliability diagram (predicted vs actual probability)
