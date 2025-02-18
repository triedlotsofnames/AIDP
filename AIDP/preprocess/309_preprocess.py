#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import argparse

def compute_family_size(df):
    """Compute Family_Size as the sum of SibSp and Parch."""
    df['Family_Size'] = df['SibSp'] + df['Parch']
    return df

def impute_embarked(df):
    """Fill missing Embarked values with the mode."""
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def replace_infinite(df):
    """Replace any infinite values with NaN."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def one_hot_encode(df):
    """One-hot encode categorical columns ('Sex' and 'Embarked')."""
    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    return df

def impute_age_model(df):
    """Impute missing Age values using a RandomForestRegressor."""
    print("Training Data - Age summary BEFORE imputation:")
    print(df['Age'].describe())
    
    train_data = df[df['Age'].notnull()]
    missing_data = df[df['Age'].isnull()]
    
    feature_cols = ['Pclass', 'SibSp', 'Parch', 'Fare']
    feature_cols += [col for col in df.columns if col.startswith('Sex_') or col.startswith('Embarked_')]
    
    X_train = train_data[feature_cols]
    y_train = train_data['Age']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    if not missing_data.empty:
        X_missing = missing_data[feature_cols]
        predicted_ages = model.predict(X_missing)
        df.loc[df['Age'].isnull(), 'Age'] = predicted_ages
    
    print("Missing Age values after imputation:", df['Age'].isnull().sum())
    print("Training Data - Age summary AFTER imputation:")
    print(df['Age'].describe())
    
    return model, df

def impute_age_with_model(df, model):
    """Impute missing Age values in test data using the provided model."""
    feature_cols = ['Pclass', 'SibSp', 'Parch', 'Fare']
    feature_cols += [col for col in df.columns if col.startswith('Sex_') or col.startswith('Embarked_')]
    
    missing_data = df[df['Age'].isnull()]
    if not missing_data.empty:
        X_missing = missing_data[feature_cols]
        predicted_ages = model.predict(X_missing)
        df.loc[df['Age'].isnull(), 'Age'] = predicted_ages
    return df

def fill_missing_numerical(df):
    """Fill missing numerical values with median."""
    numerical_cols = ['Fare', 'Family_Size']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

def add_age_group(df):
    """Create an Age_Group column based on specified bins."""
    bins = [0, 12, 18, 30, 50, 80]
    labels = ['0-12', '13-18', '19-30', '31-50', '51-80']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df

def add_fare_log(df):
    """Add a Fare_log column by applying log1p transformation to Fare."""
    df['Fare_log'] = np.log1p(df['Fare'])
    return df

def drop_unwanted_columns(df):
    """Drop columns that are not needed for further analysis."""
    columns_to_drop = ["Cabin", "Ticket", "PassengerId", "Name", "SibSp", "Parch", "Fare"]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

def scale_features(train_df, test_df, features_to_scale):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
    return train_df, test_df

def preprocess_train_test(train_df, test_df):
    """Preprocess both the training and test datasets."""
    train_df = compute_family_size(train_df)
    test_df = compute_family_size(test_df)
    
    train_df = impute_embarked(train_df)
    test_df = impute_embarked(test_df)
    
    train_df = replace_infinite(train_df)
    test_df = replace_infinite(test_df)
    
    train_df = fill_missing_numerical(train_df)
    test_df = fill_missing_numerical(test_df)
    
    train_df = one_hot_encode(train_df)
    test_df = one_hot_encode(test_df)
    
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined = pd.concat([train_df, test_df], sort=False)
    combined = one_hot_encode(combined)
    train_df = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test_df = combined[combined['is_train'] == 0].drop(columns=['is_train'])
    
    age_model, train_df = impute_age_model(train_df)
    test_df = impute_age_with_model(test_df, age_model)
    
    train_df = add_age_group(train_df)
    test_df = add_age_group(test_df)
    
    train_df = add_fare_log(train_df)
    test_df = add_fare_log(test_df)
    
    train_df = drop_unwanted_columns(train_df)
    test_df = drop_unwanted_columns(test_df)
    
    features_to_scale = ['Age', 'Fare_log', 'Family_Size']
    train_df, test_df = scale_features(train_df, test_df, features_to_scale)

    print("\nFinal preprocessed training data preview:")
    print(train_df.head())
    print("\nFinal preprocessed test data preview:")
    print(test_df.head())
    
    return train_df, test_df

def main(train_input, test_input, train_output, test_output):
    """Load, preprocess, and save training and test datasets."""
    print("Loading training data from:", train_input)
    train_df = pd.read_csv(train_input)
    print("Loading test data from:", test_input)
    test_df = pd.read_csv(test_input)
    
    preprocessed_train, preprocessed_test = preprocess_train_test(train_df, test_df)

    # Ensure no missing values in the final test dataset
    preprocessed_test = preprocessed_test.fillna(method='ffill')  # Forward fill as final safeguard

    print("\nFinal preprocessed test dataset preview before saving:")
    print(preprocessed_test.head())

    preprocessed_train.to_csv(train_output, index=False)
    preprocessed_test.to_csv(test_output, index=False)

    print(f"\nPreprocessed training data saved to {train_output}")
    print(f"Preprocessed test data saved to {test_output} (missing values filled)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Titanic Data Preprocessing Pipeline")
    parser.add_argument('--train', type=str, default='data/train.csv', help="Path to the training CSV file")
    parser.add_argument('--test', type=str, default='data/test.csv', help="Path to the test CSV file")
    parser.add_argument('--train_output', type=str, default='data/preprocessed_train.csv', help="Path to preprocessed training CSV file")
    parser.add_argument('--test_output', type=str, default='data/preprocessed_test.csv', help="Path to preprocessed test CSV file")
    args = parser.parse_args()
    
    main(args.train, args.test, args.train_output, args.test_output)
