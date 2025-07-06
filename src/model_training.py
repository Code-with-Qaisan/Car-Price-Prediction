import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    """
    A class to handle car price prediction using traditional ML models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train traditional machine learning models.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            
        Returns:
            dict: Dictionary containing trained models and their results
        """
        # Define models (matching the notebook implementation)
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics (matching notebook metrics)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
        
        self.models.update(results)
        return results
    
    def get_best_model(self):
        """
        Get the best performing model based on R² score.
        
        Returns:
            tuple: (best_model_name, best_model_object)
        """
        best_model_name = None
        best_model_obj = None
        best_r2 = -float('inf')
        
        for name, results in self.models.items():
            if results['R2'] > best_r2:
                best_r2 = results['R2']
                best_model_name = name
                best_model_obj = results['model']
        
        return best_model_name, best_model_obj
    
    def predict_with_best_model(self, X):
        """
        Make predictions using the best performing model.
        
        Args:
            X: Input features
            
        Returns:
            array: Predictions
        """
        best_model_name, best_model_results = self.get_best_model()
        model = best_model_results['model']
        
        return model.predict(X)
    
    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame for easy visualization.
        
        Returns:
            pandas.DataFrame: Results dataframe
        """
        results_dict = {}
        for name, results in self.models.items():
            results_dict[name] = {
                'RMSE': results['RMSE'],
                'MAE': results['MAE'],
                'R2': results['R2']
            }
        
        return pd.DataFrame(results_dict).T 