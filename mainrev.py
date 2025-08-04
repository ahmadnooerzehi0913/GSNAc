import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from dataset_handler_cleaned import DatasetHandler
from classifier import GSNAcClassifier

def main(cv_value, dataset_name, random_state, n_neighbors, ac_type):
    print(f"Dataset: {dataset_name}")

    # بارگذاری و پیش‌پردازش دیتا
    dh = DatasetHandler('iris')
    X, y, processed_df, categorical_feature_names = dh.handler()
    
    # اطمینان از اینکه indexهای X و y تنظیم‌اند
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    
    skf = StratifiedKFold(n_splits=cv_value, shuffle=True, random_state=random_state)
    all_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")
        std_scale = StandardScaler()
        X.iloc[train_idx] = std_scale.fit_transform(X.iloc[train_idx])
        X.iloc[test_idx] = std_scale.transform(X.iloc[test_idx])

        clf = GSNAcClassifier(n_neighbors=n_neighbors, ac_type=ac_type)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx].ravel())
        y_pred = clf.predict(X.iloc[test_idx])
        acc = accuracy_score(y.iloc[test_idx], y_pred)
        print(f"Accuracy: {acc:.4f}")
        all_scores.append(acc)

    print("\n--- Summary ---")
    print(f"Mean Accuracy: {np.mean(all_scores):.4f}")
    print(f"Std Dev: {np.std(all_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_value', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, choices=['iris', 'titanic', 'pima'], required=True)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--ac_type', type=str, default='e', help='e for Euclidean, m for Manhattan, c for Cosine')

    args = parser.parse_args()
    main(args.cv_value, args.dataset_name, args.random_state, args.n_neighbors, args.ac_type)
