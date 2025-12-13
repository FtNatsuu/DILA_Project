import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import jellyfish

##################################################################################
#                                 Configuration                                  #
##################################################################################

SIM_FILE = "scores/similarity_scores.csv"
MAPPING_FILE = "DBLP-Scholar_perfectMapping.csv"
DBLP_FILE = "DBLP1.csv"
SCHOLAR_FILE = "Scholar.csv"

RANDOM_STATE = 42


##################################################################################
#                            Create Training Dataset                             #
##################################################################################

# For each pair in sim_df we check if the (DBLP, Scholar) IDs exist in the ground truth set gt_set
# If so, we assign a label 1, otherwise 0
# We add the label column to sim_df

print("\n2.1: Creating training dataset...")

sim_df = pd.read_csv(SIM_FILE)
gt = pd.read_csv(MAPPING_FILE)
gt_set = set(zip(gt['idDBLP'], gt['idScholar']))                                # ground truth pairs

labels = []

for _, row in sim_df.iterrows():
    dblp_id = row['id_dblp']
    scholar_id = row['id_scholar']

    if (dblp_id, scholar_id) in gt_set:
        labels.append(1)   # Positive match
    else:
        labels.append(0)   # Negative match

sim_df['label'] = labels                                                        # Assign labels to the DataFrame

pos = sim_df['label'].sum()
neg = len(sim_df) - pos

print(f"Total pairs: {len(sim_df)} | Positive: {pos} ({pos/len(sim_df)*100:.2f}%) | Negative: {neg}")


##################################################################################
#                              Feature Engineering                               #
##################################################################################

# For each article pair in sim_df, we retrieve their full records from DBLP and Scholar
# Then compute several similarity measures
# These new features are appended to sim_df, providing all similarity metrics ready for analysis or model training

print("\n2.1: Feature Engineering...")

dblp = pd.read_csv(DBLP_FILE, encoding='latin-1')
scholar = pd.read_csv(SCHOLAR_FILE, encoding='utf-8')

dblp_dict = {f"DBLP_{i}": r for i, r in dblp.iterrows()}                        # Create dictionaries for fast lookup (the key is the ID and the value is the record)
scholar_dict = {f"SCH_{i}": r for i, r in scholar.iterrows()}

features = []
for i, row in sim_df.iterrows():
    rec_d = dblp_dict.get(row['id_dblp'], {})                                   # Retrieve records
    rec_s = scholar_dict.get(row['id_scholar'], {})
    
    title_d = str(rec_d.get('title', '')).lower()
    title_s = str(rec_s.get('title', '')).lower()
    auth_d = str(rec_d.get('authors', '')).lower()
    auth_s = str(rec_s.get('authors', '')).lower()
    venue_d = str(rec_d.get('venue', '')).lower()
    venue_s = str(rec_s.get('venue', '')).lower()
    
    # Additional features
    len_diff = abs(len(title_d) - len(title_s)) / max(len(title_d), len(title_s), 1)
    tokens_d = set(title_d.split())
    tokens_s = set(title_s.split())
    shared_tokens = len(tokens_d.intersection(tokens_s))
    chars_d = set(title_d.replace(' ', ''))
    chars_s = set(title_s.replace(' ', ''))
    char_overlap = len(chars_d.intersection(chars_s)) / max(len(chars_d), len(chars_s), 1)
    venue_exact = 1.0 if venue_d and venue_d == venue_s else 0.0
    auth_d_set = set(auth_d.split())
    auth_s_set = set(auth_s.split())
    auth_overlap = len(auth_d_set.intersection(auth_s_set)) / max(len(auth_d_set), len(auth_s_set), 1)
    jaro = jellyfish.jaro_winkler_similarity(title_d, title_s) if title_d and title_s else 0
    
    features.append([len_diff, shared_tokens, char_overlap, venue_exact, auth_overlap, jaro])
    
    if (i+1) % 500000 == 0:
        print(f"  Processed {i+1}/{len(sim_df)} pairs...")

# Append new features to DataFrame
feat_df = pd.DataFrame(features, columns=['title_len_diff', 'shared_tokens', 'char_overlap', 'venue_exact', 'auth_overlap', 'jaro_winkler'])

# Combine with existing similarity features
sim_df = pd.concat([sim_df.reset_index(drop=True), feat_df], axis=1)
sim_df.head()

feature_cols = ['cosine', 'jaccard', 'dice', 'levenshtein', 'author_jaccard', 'venue_jaccard', 'year_match', 'title_len_diff', 'shared_tokens', 'char_overlap', 'venue_exact', 'auth_overlap', 'jaro_winkler']
print(f"Total features: {len(feature_cols)}")


##################################################################################
#                             Process Training Data                              #
##################################################################################

# Fill missing values and split features and labels 
# Standardize the features  and apply undersampling to balance the classes

print("\n2.3: Preprocessing data...")

sim_df[feature_cols] = sim_df[feature_cols].fillna(0)                       # Handle missing values

# Split features and labels
X = sim_df[feature_cols].values
y = sim_df['label'].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance - because we have a really unbalanced dataset
print("Applying undersampling to balance classes...")
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
    
# Sample same number of negatives as positives
neg_sample = np.random.choice(neg_idx, size=len(pos_idx), replace=False)        # pick  as many random negatives as we have positives
balanced_idx = np.concatenate([pos_idx, neg_sample])                            # combine indices
np.random.shuffle(balanced_idx)

# Keep only balanced samples
X_balanced = X_scaled[balanced_idx]
y_balanced = y[balanced_idx]
print(f"Balanced dataset: {len(X_balanced)} samples (50% positive)")


##################################################################################
#                                 Model Training                                 #
##################################################################################

# Train and evaluate models using 3-fold cross-validation
# Train the final model on the entire balanced dataset

print("\n2.4: Training model...")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE) # Split data in 3 folds for cross-validation

# Define models with tuned hyperparameters
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
}

results = {}
for name, model in models.items():
    print(f"\n{name}:")
    
    # Cross-validation
    f1_scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='f1', n_jobs=-1)
    prec_scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='precision', n_jobs=-1)
    rec_scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='recall', n_jobs=-1)
    
    print(f"  Precision: {prec_scores.mean()} (+/- {prec_scores.std()*2})")
    print(f"  Recall:    {rec_scores.mean()} (+/- {rec_scores.std()*2})")
    print(f"  F1-Score:  {f1_scores.mean()} (+/- {f1_scores.std()*2})")
    
    results[name] = {'f1': f1_scores.mean(), 'precision': prec_scores.mean(), 'recall': rec_scores.mean()}
    
    # Train final model
    model.fit(X_balanced, y_balanced)


##################################################################################
#                                  Evaluation                                    #
##################################################################################

print("2.5: Evaluating models...")

baselines = {'SVM': 0.76, 'Random Forest': 0.79}                    # Baseline F1-scores

# Compare with baselines
summary = []
for name, metrics in results.items():
    baseline = baselines[name]
    improvement = (metrics['f1'] - baseline) / baseline * 100
    beats = "YES" if metrics['f1'] > baseline else "NO"
    
    print(f"\n{name}:")
    print(f"  F1-Score:    {metrics['f1']}")
    print(f"  Baseline:    {baseline}")
    print(f"  Improvement: {improvement:+.2f}%")
    print(f"  Beats baseline: {beats}")
    
    summary.append({
        'Model': name,
        'F1': metrics['f1'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'Baseline': baseline,
        'Beats': beats,
        'Improvement_%': improvement
    })

# Feature importance for Random Forest
rf_model = models['Random Forest']

importance_df = pd.DataFrame({'Feature': feature_cols,'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)

print("Top 5 Feature Importances random forest:")
for i, row in importance_df.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']}")

# Save results
summary_df = pd.DataFrame(summary)
summary_df.to_csv('eval/model_evaluation.csv', index=False)
importance_df.to_csv('eval/feature_importance.csv', index=False)

print("\nFiles saved: eval/model_evaluation.csv, eval/feature_importance.csv")