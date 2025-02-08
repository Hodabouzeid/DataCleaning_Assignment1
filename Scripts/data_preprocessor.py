# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Global variables to track changes
removed_rows_missing = 0
removed_cols_missing = 0
removed_features_redundant = 0

def impute_missing_values(data, strategy='mean'):
    """ Fill missing values and track removed columns. """
    global removed_cols_missing  # Track columns removed due to missing values
    
    print("\n Step 1: Imputing Missing Values... Function Called!", flush=True)
    
    if data is None or data.empty:
        print(" ERROR: The input dataset is None or empty.", flush=True)
        return None
    
    print(f"Input Data Shape Before Imputation: {data.shape}", flush=True)
    
    # Convert non-numeric values to NaN before imputation
    data = data.copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    print(f" Data Shape After Conversion: {data.shape}", flush=True)
    
    # Apply imputation strategy
    if strategy == 'mean':
        data.fillna(data.mean(numeric_only=True), inplace=True)
    elif strategy == 'median':
        data.fillna(data.median(numeric_only=True), inplace=True)
    elif strategy == 'mode':
        data.fillna(data.mode().iloc[0], inplace=True)
    else:
        raise ValueError("Invalid strategy! Choose 'mean', 'median', or 'mode'.")
    
    # Count columns that are still entirely NaN and remove them
    nan_columns = data.columns[data.isna().all()]
    removed_cols_missing = len(nan_columns)
    
    if removed_cols_missing > 0:
        print(f"Removing {removed_cols_missing} columns with all NaN values: {list(nan_columns)}", flush=True)
        data.drop(columns=nan_columns, inplace=True)
    
    print(f" Data Shape After Imputation & NaN Column Removal: {data.shape}", flush=True)
    return data


def remove_duplicates(data):
    """ Remove duplicate rows and track how many were removed. """
    global removed_rows_missing  # Track rows removed due to duplication
    
    print("\n Step 2: Removing Duplicate Rows...", flush=True)
    
    before_rows = data.shape[0]
    data = data.drop_duplicates()
    removed_rows_missing = before_rows - data.shape[0]
    
    print(f" {removed_rows_missing} duplicate rows removed.", flush=True)
    return data


def normalize_data(data, method='minmax'):
    """ Normalize numerical features. """
    print("\n Step 3: Normalizing Numerical Data...", flush=True)
    
    numeric_columns = data.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data


def remove_redundant_features(data, threshold=0.9):
    """ Remove redundant features based on correlation. """
    global removed_features_redundant  # Track removed features
    
    print("\n Step 4: Removing Redundant Features...", flush=True)
    
    correlation_matrix = data.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    removed_features_redundant = len(to_drop)
    
    print(f" Dropping {removed_features_redundant} redundant features: {to_drop}", flush=True)
    return data.drop(columns=to_drop)


# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """ Train logistic regression model and track accuracy. """
    print("\n Running Simple Model...", flush=True)

    if input_data.shape[0] == 0 or input_data.shape[1] == 0:
        print(" ERROR: Empty dataset. Cannot train the model.", flush=True)
        return None
    
    target = input_data[input_data.columns[0]]
    features = input_data[input_data.columns[1:]]

    if target.dtype == 'float64' or target.dtype == 'float32':
        print(f" WARNING: Target column '{input_data.columns[0]}' is continuous! Converting to integer labels.", flush=True)
        target = target.fillna(target.mode()[0])  # Fill missing target values
        target = target.round().astype(int)  # Convert to categorical (integer values)

    if target.nunique() < 2:
        print(" ERROR: Target column must have at least two classes!", flush=True)
        return None
    
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    
    if scale_data:
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=100, solver='liblinear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n Model Accuracy: {accuracy:.2f}", flush=True)

    if print_report:
        print("\n Classification Report:", flush=True)
        print(classification_report(y_test, y_pred), flush=True)

    return accuracy  # Return accuracy to compare later
