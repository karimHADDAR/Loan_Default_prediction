""" Training script for loan default prediction model. """

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb

# Import utility functions
from utils import (
    load_and_filter_data,
    create_target,
    feature_engineering_pipeline,
    handle_missing_values,
    encode_categoricals,
    get_important_features,
    calculate_class_weight,
    print_data_summary
)


def prepare_data(data_path: str, output_path: str = None) -> tuple:
    """
    Load and prepare data for modeling.
    
    Args:
        data_path: Path to raw data CSV
        output_path: Optional path to save processed data
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, target_encodings)
    """
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*70 + "\n")
    
    # Load data
    df = load_and_filter_data(data_path, completed_only=True)
    
    # Create target
    df = create_target(df)
    
    # Feature engineering
    df = feature_engineering_pipeline(df)
    
    # Select important features
    important_features = get_important_features()
    available_features = [f for f in important_features if f in df.columns]
    available_features.append('default')  # Add target
    
    print(f"Using {len(available_features)-1} important features")
    df = df[available_features].copy()
    
    # Handle missing values
    df = handle_missing_values(df, threshold=0.5)
    
    # Print summary
    print_data_summary(df, "Processed Data")
    
    # Separate features and target
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Encode categorical variables
    X, target_encodings = encode_categoricals(X, y, high_card_threshold=10)
    
    # Train-test split
    print("\nSplitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train default rate: {y_train.mean()*100:.2f}%")
    print(f"Test default rate: {y_test.mean()*100:.2f}%")
    
    # Optionally save processed data
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to {output_path}")
    
    return X_train, X_test, y_train, y_test, target_encodings


def train_baseline(X_train, y_train, X_test, y_test):
    """Train baseline logistic regression model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    
    print("\n" + "="*70)
    print("STEP 2: BASELINE MODEL (Logistic Regression)")
    print("="*70 + "\n")
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_pred = lr_model.predict(X_test)
    
    lr_auc = roc_auc_score(y_test, lr_pred_proba)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    print(f"Baseline Performance:")
    print(f"  ROC-AUC: {lr_auc:.4f}")
    print(f"  F1 Score: {lr_f1:.4f}")
    print(f"  Accuracy: {lr_acc:.4f}")
    
    return lr_auc, lr_f1, lr_acc


def train_xgboost_initial(X_train, y_train, X_test, y_test):
    """Train initial XGBoost model."""
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    
    print("\n" + "="*70)
    print("STEP 3: INITIAL XGBOOST MODEL")
    print("="*70 + "\n")
    
    # Calculate class weight
    scale_pos_weight = calculate_class_weight(y_train)
    
    # Train model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='auc'
    )
    
    print("Training XGBoost model...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    xgb_f1 = f1_score(y_test, xgb_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print(f"\nInitial XGBoost Performance:")
    print(f"  ROC-AUC: {xgb_auc:.4f}")
    print(f"  F1 Score: {xgb_f1:.4f}")
    print(f"  Accuracy: {xgb_acc:.4f}")
    
    return xgb_model, xgb_auc, xgb_f1, xgb_acc


def tune_hyperparameters(X_train, y_train, scale_pos_weight, n_iter=20):
    """Tune XGBoost hyperparameters."""
    print("\n" + "="*70)
    print("STEP 4: HYPERPARAMETER TUNING")
    print("="*70 + "\n")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    
    print(f"Running RandomizedSearchCV with {n_iter} iterations...")
    print("This may take several minutes...\n")
    
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Tuning complete!")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def evaluate_final_model(model, X_test, y_test):
    """Evaluate final model comprehensively."""
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        precision_score, recall_score, average_precision_score,
        classification_report
    )
    
    print("\n" + "="*70)
    print("STEP 5: FINAL MODEL EVALUATION")
    print("="*70 + "\n")
    
    final_pred = model.predict(X_test)
    final_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, final_pred_proba),
        'f1_score': f1_score(y_test, final_pred),
        'accuracy': accuracy_score(y_test, final_pred),
        'precision': precision_score(y_test, final_pred),
        'recall': recall_score(y_test, final_pred),
        'average_precision': average_precision_score(y_test, final_pred_proba)
    }
    
    print("🏆 FINAL MODEL PERFORMANCE")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")
    print("="*50)
    
    print(f"\nClassification Report:\n{classification_report(y_test, final_pred)}")
    
    return metrics


def save_model_artifacts(model, feature_names, target_encodings, metrics, best_params, output_dir='../models'):
    """Save model and related artifacts."""
    print("\n" + "="*70)
    print("STEP 6: SAVING MODEL ARTIFACTS")
    print("="*70 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save model
    model_path = output_path / 'xgboost_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save feature names
    features_path = output_path / 'feature_names.pkl'
    joblib.dump(feature_names, features_path)
    print(f"✅ Feature names saved: {features_path}")
    
    # Save target encodings
    encodings_path = output_path / 'target_encodings.pkl'
    joblib.dump(target_encodings, encodings_path)
    print(f"✅ Target encodings saved: {encodings_path}")
    
    # Save metrics
    metrics['best_params'] = best_params
    metrics['trained_at'] = datetime.now().isoformat()
    metrics_path = output_path / 'model_metrics.pkl'
    joblib.dump(metrics, metrics_path)
    print(f"✅ Metrics saved: {metrics_path}")
    
    print(f"\n✅ All artifacts saved to {output_path}")


def main(args):
    """Main training pipeline."""
    print("\n" + "="*70)
    print("LOAN DEFAULT PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Prepare data
    X_train, X_test, y_train, y_test, target_encodings = prepare_data(
        args.data_path,
        args.processed_data_path
    )
    
    # Step 2: Train baseline
    lr_auc, lr_f1, lr_acc = train_baseline(X_train, y_train, X_test, y_test)
    
    # Step 3: Train initial XGBoost
    xgb_initial, xgb_auc, xgb_f1, xgb_acc = train_xgboost_initial(
        X_train, y_train, X_test, y_test
    )
    
    # Step 4: Tune hyperparameters
    scale_pos_weight = calculate_class_weight(y_train)
    best_model, best_params = tune_hyperparameters(
        X_train, y_train, scale_pos_weight, args.n_iter
    )
    
    # Step 5: Evaluate final model
    metrics = evaluate_final_model(best_model, X_test, y_test)
    
    # Step 6: Save artifacts
    save_model_artifacts(
        best_model,
        X_train.columns.tolist(),
        target_encodings,
        metrics,
        best_params,
        args.model_dir
    )
    
    print("\n" + "="*70)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFinal ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Improvement over baseline: {(metrics['roc_auc'] - lr_auc)*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train loan default prediction model')
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data/accepted_2007_to_2018Q4.csv',
        help='Path to raw data CSV file'
    )
    parser.add_argument(
        '--processed-data-path',
        type=str,
        default='../data/processed.csv',
        help='Path to save processed data'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='../models',
        help='Directory to save model artifacts'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=20,
        help='Number of iterations for hyperparameter tuning'
    )
    
    args = parser.parse_args()
    main(args)
