"""
Utility functions for loan default prediction project.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def load_and_filter_data(filepath: str, completed_only: bool = True) -> pd.DataFrame:
    """
    Load loan data and optionally filter to completed loans only.
    
    Args:
        filepath: Path to CSV file
        completed_only: If True, filter to only Fully Paid and Charged Off loans
        
    Returns:
        Filtered DataFrame
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records with {df.shape[1]} features")
    
    if completed_only and 'loan_status' in df.columns:
        df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
        print(f"Filtered to {len(df):,} completed loans")
    
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable for loan default.
    
    Args:
        df: DataFrame with loan_status column
        
    Returns:
        DataFrame with new 'default' column
    """
    default_status = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
    df['default'] = df['loan_status'].isin(default_status).astype(int)
    
    print(f"Target created - Default rate: {df['default'].mean()*100:.2f}%")
    return df


def parse_interest_rate(df: pd.DataFrame, col: str = 'int_rate') -> pd.DataFrame:
    """Parse interest rate from string to float."""
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.rstrip('%').astype('float')
    return df


def parse_term(df: pd.DataFrame, col: str = 'term') -> pd.DataFrame:
    """Parse term from string to number of months."""
    if col in df.columns:
        df['term_months'] = df[col].str.extract('(\d+)').astype('float')
        df.drop(col, axis=1, inplace=True)
    return df


def parse_employment_length(df: pd.DataFrame, col: str = 'emp_length') -> pd.DataFrame:
    """Parse employment length to years."""
    if col in df.columns:
        emp_length_map = {
            '< 1 year': 0,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10
        }
        df['emp_length_years'] = df[col].map(emp_length_map)
        df.drop(col, axis=1, inplace=True)
    return df


def parse_credit_history(df: pd.DataFrame, col: str = 'earliest_cr_line') -> pd.DataFrame:
    """Calculate credit history length in years."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce')
        df['credit_history_years'] = (pd.Timestamp.now() - df[col]).dt.days / 365.25
        df.drop(col, axis=1, inplace=True)
    return df


def parse_revolving_util(df: pd.DataFrame, col: str = 'revol_util') -> pd.DataFrame:
    """Parse revolving utilization from string to float."""
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.rstrip('%').astype('float')
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new derived features
    """
    # Loan to income ratio
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    
    # Installment to income ratio (annualized)
    if 'installment' in df.columns and 'annual_inc' in df.columns:
        df['installment_to_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)
    
    print(f"Created derived features. New shape: {df.shape}")
    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Engineered DataFrame
    """
    print("\n=== Starting Feature Engineering ===")
    
    # Parse string columns
    df = parse_interest_rate(df)
    df = parse_term(df)
    df = parse_employment_length(df)
    df = parse_credit_history(df)
    df = parse_revolving_util(df)
    
    # Create derived features
    df = create_derived_features(df)
    
    print("=== Feature Engineering Complete ===\n")
    return df


def handle_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values by dropping high-missing columns and imputing others.
    
    Args:
        df: Input DataFrame
        threshold: Drop columns with missing rate above this threshold
        
    Returns:
        DataFrame with handled missing values
    """
    # Drop high-missing columns
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing")
        df.drop(cols_to_drop, axis=1, inplace=True)
    
    # Impute numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Impute categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"Missing values handled. Remaining: {df.isnull().sum().sum()}")
    return df


def encode_categoricals(
    X: pd.DataFrame, 
    y: pd.Series,
    high_card_threshold: int = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.
    High-cardinality: target encoding
    Low-cardinality: one-hot encoding
    
    Args:
        X: Feature DataFrame
        y: Target Series
        high_card_threshold: Threshold for determining high cardinality
        
    Returns:
        Encoded DataFrame and dictionary of target encodings
    """
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Separate by cardinality
    high_card_cols = [col for col in categorical_cols if X[col].nunique() > high_card_threshold]
    low_card_cols = [col for col in categorical_cols if X[col].nunique() <= high_card_threshold]
    
    print(f"High cardinality columns (>{high_card_threshold}): {high_card_cols}")
    print(f"Low cardinality columns (<={high_card_threshold}): {low_card_cols}")
    
    # Target encoding for high cardinality
    target_encodings = {}
    for col in high_card_cols:
        # Combine X and y temporarily for groupby
        temp_df = pd.concat([X[col], y], axis=1)
        encoding = temp_df.groupby(col)[y.name].mean()
        target_encodings[col] = encoding
        X[col] = X[col].map(encoding)
        X[col].fillna(y.mean(), inplace=True)
    
    # One-hot encoding for low cardinality
    if low_card_cols:
        X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)
    
    print(f"Encoding complete. Final shape: {X.shape}")
    return X, target_encodings


def get_important_features() -> List[str]:
    """
    Return list of important features for credit risk modeling.
    
    Returns:
        List of feature names
    """
    return [
        # Loan characteristics
        'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'term',
        'grade', 'sub_grade', 'purpose', 'home_ownership',
        
        # Borrower profile
        'annual_inc', 'emp_length', 'verification_status',
        
        # Credit history
        'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
        'revol_util', 'total_acc', 'delinq_2yrs', 'inq_last_6mths',
        'mort_acc', 'pub_rec_bankruptcies',
        
        # Geographic
        'addr_state'
    ]


def calculate_class_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost to handle imbalanced data.
    
    Args:
        y: Target Series
        
    Returns:
        Scale pos weight value
    """
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    return scale_pos_weight


def print_data_summary(df: pd.DataFrame, name: str = "Data"):
    """Print summary statistics for a DataFrame."""
    print(f"\n{'='*50}")
    print(f"{name} Summary")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing Values: {df.isnull().sum().sum()}")
    
    if 'default' in df.columns:
        print(f"\nDefault Rate: {df['default'].mean()*100:.2f}%")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test utility functions
    print("Utility Functions Module")
    print("Import this module to use these functions in your pipeline")
