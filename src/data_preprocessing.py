import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    Load and perform initial cleaning on the car price dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Standardize car names
    car_name_replacements = {
        'maxda': 'mazda',
        'porcshce': 'porsche',
        'toyouta': 'toyota',
        'vokswagen': 'volkswagen',
        'vw': 'volkswagen'
    }
    
    df['CarName'] = df['CarName'].replace(car_name_replacements)
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with handled missing values
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        
        # Handle missing values based on data type
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype == 'object':
                    # For categorical variables, fill with mode
                    df[column].fillna(df[column].mode()[0], inplace=True)
                else:
                    # For numerical variables, fill with median
                    df[column].fillna(df[column].median(), inplace=True)
    
    return df

def encode_categorical_variables(df, categorical_columns):
    """
    Encode categorical variables using label encoding.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        categorical_columns (list): List of categorical column names
        
    Returns:
        pandas.DataFrame: Dataframe with encoded categorical variables
    """
    df_encoded = df.copy()
    label_encoders = {}
    
    for column in categorical_columns:
        if column in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            label_encoders[column] = le
    
    return df_encoded, label_encoders

def scale_numerical_features(df, numerical_columns, scaler=None):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        numerical_columns (list): List of numerical column names
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        tuple: (scaled_dataframe, fitted_scaler)
    """
    df_scaled = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
    else:
        df_scaled[numerical_columns] = scaler.transform(df_scaled[numerical_columns])
    
    return df_scaled, scaler

def prepare_features(df, target_column='price'):
    """
    Prepare features for modeling by separating features and target.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X_features, y_target)
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = handle_missing_values(df_processed)
    
    # Identify categorical and numerical columns
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target column from numerical columns if present
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    # Encode categorical variables
    if categorical_columns:
        df_processed, _ = encode_categorical_variables(df_processed, categorical_columns)
    
    # Separate features and target
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    return X, y 