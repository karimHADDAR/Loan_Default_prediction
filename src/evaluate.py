"""
Evaluation script for loan default prediction model.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, accuracy_score, precision_score, recall_score
)


def load_model_artifacts(model_dir='../models'):
    """Load trained model and artifacts."""
    model_path = Path(model_dir)
    
    print("Loading model artifacts...")
    model = joblib.load(model_path / 'xgboost_model.pkl')
    feature_names = joblib.load(model_path / 'feature_names.pkl')
    metrics = joblib.load(model_path / 'model_metrics.pkl')
    
    print(f"✅ Model loaded from {model_path}")
    print(f"Features: {len(feature_names)}")
    print(f"Training ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return model, feature_names, metrics


def evaluate_model(model, X_test, y_test, output_dir='../models'):
    """
    Comprehensive model evaluation with visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save evaluation plots
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    # Print metrics
    print("📊 Performance Metrics:")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")
    print("="*50)
    
    # Classification report
    print(f"\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Fully Paid', 'Defaulted']))
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 Confusion Matrix Breakdown:")
    print(f"True Negatives (Correctly predicted Fully Paid): {tn:,}")
    print(f"False Positives (Predicted Default, Actually Paid): {fp:,}")
    print(f"False Negatives (Predicted Paid, Actually Defaulted): {fn:,}")
    print(f"True Positives (Correctly predicted Default): {tp:,}")
    
    print(f"\n💡 Business Insights:")
    print(f"Model catches {tp/(tp+fn)*100:.1f}% of actual defaults")
    print(f"False alarm rate: {fp/(fp+tn)*100:.1f}%")
    
    # Generate visualizations
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. ROC Curve
    plot_roc_curve(y_test, y_pred_proba, output_path)
    
    # 2. Precision-Recall Curve
    plot_pr_curve(y_test, y_pred_proba, output_path)
    
    # 3. Confusion Matrix
    plot_confusion_matrix(cm, output_path)
    
    # 4. Feature Importance
    plot_feature_importance(model, X_test.columns, output_path)
    
    # 5. Prediction Distribution
    plot_prediction_distribution(y_test, y_pred_proba, output_path)
    
    return metrics


def plot_roc_curve(y_test, y_pred_proba, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_value = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Loan Default Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = output_path / 'roc_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved: {save_path}")


def plot_pr_curve(y_test, y_pred_proba, output_path):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {pr_auc:.4f})')
    plt.axhline(y=y_test.mean(), color='navy', linestyle='--', 
                label=f'Baseline (Default Rate = {y_test.mean():.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = output_path / 'precision_recall_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PR curve saved: {save_path}")


def plot_confusion_matrix(cm, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Fully Paid', 'Defaulted'],
                yticklabels=['Fully Paid', 'Defaulted'],
                annot_kws={'fontsize': 14})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_path / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


def plot_feature_importance(model, feature_names, output_path, top_n=20):
    """Plot and save feature importance."""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 10))
    top_features = feature_importance.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    save_path = output_path / 'feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance plot saved: {save_path}")
    
    # Also save as CSV
    csv_path = output_path / 'feature_importance.csv'
    feature_importance.to_csv(csv_path, index=False)
    print(f"✅ Feature importance CSV saved: {csv_path}")


def plot_prediction_distribution(y_test, y_pred_proba, output_path):
    """Plot distribution of predicted probabilities by true label."""
    plt.figure(figsize=(12, 6))
    
    # Predictions for each class
    paid_probs = y_pred_proba[y_test == 0]
    default_probs = y_pred_proba[y_test == 1]
    
    plt.hist(paid_probs, bins=50, alpha=0.6, label='Fully Paid', color='green', density=True)
    plt.hist(default_probs, bins=50, alpha=0.6, label='Defaulted', color='red', density=True)
    
    plt.xlabel('Predicted Default Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = output_path / 'prediction_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Prediction distribution plot saved: {save_path}")


def analyze_threshold_impact(y_test, y_pred_proba, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Analyze impact of different decision thresholds."""
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS")
    print("="*70 + "\n")
    
    results = []
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'Threshold': threshold,
            'Precision': precision_score(y_test, y_pred_thresh),
            'Recall': recall_score(y_test, y_pred_thresh),
            'F1': f1_score(y_test, y_pred_thresh),
            'Accuracy': accuracy_score(y_test, y_pred_thresh)
        }
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    print("📊 Threshold Impact Analysis:")
    print(df_results.to_string(index=False))
    
    print("\n💡 Business Interpretation:")
    print("- Lower threshold: Catch more defaults (higher recall) but more false alarms")
    print("- Higher threshold: Fewer false alarms (higher precision) but miss more defaults")
    print("- Choose threshold based on business objectives and costs")
    
    return df_results


def main(args):
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("LOAN DEFAULT PREDICTION - MODEL EVALUATION")
    print("="*70)
    
    # Load model
    model, feature_names, training_metrics = load_model_artifacts(args.model_dir)
    
    # Load test data
    print(f"\nLoading test data from {args.test_data_path}...")
    # Note: In production, you'd apply the same preprocessing pipeline
    # For now, assuming processed data is available
    df_test = pd.read_csv(args.test_data_path)
    
    X_test = df_test[feature_names]
    y_test = df_test['default']
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test default rate: {y_test.mean()*100:.2f}%")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, args.output_dir)
    
    # Threshold analysis
    if args.threshold_analysis:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        threshold_results = analyze_threshold_impact(y_test, y_pred_proba)
        
        # Save results
        threshold_path = Path(args.output_dir) / 'threshold_analysis.csv'
        threshold_results.to_csv(threshold_path, index=False)
        print(f"\n✅ Threshold analysis saved: {threshold_path}")
    
    print("\n" + "="*70)
    print("🎉 EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate loan default prediction model')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='../models',
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--test-data-path',
        type=str,
        default='../data/processed.csv',
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../models',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--threshold-analysis',
        action='store_true',
        help='Perform threshold impact analysis'
    )
    
    args = parser.parse_args()
    main(args)
