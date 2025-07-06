#!/usr/bin/env python3
"""
Car Price Prediction - Main Execution Script

This script provides a command-line interface to:
1. Train and save models
2. Run the Flask web application
3. Make predictions from command line
4. Generate reports

Usage:
    python main.py train          # Train and save models
    python main.py web            # Run Flask web app
    python main.py predict        # Make prediction from CLI
    python main.py report         # Generate analysis report
"""

import argparse
import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append('src')

from data_preprocessing import load_and_clean_data, prepare_features
from model_training import CarPricePredictor

def train_models():
    """Train models and save them"""
    print("🚗 Training Car Price Prediction Models...")
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df = load_and_clean_data('data/CarPrice_Assignment.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    feature_columns = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("🤖 Training models...")
    predictor = CarPricePredictor()
    results = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Get best model
    best_model_name, best_model = predictor.get_best_model()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    print("💾 Saving models...")
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    # Print results
    print("\n📈 Model Performance Results:")
    results_df = predictor.get_results_dataframe()
    print(results_df.to_string(index=False))
    
    print(f"\n✅ Best model ({best_model_name}) saved successfully!")
    print("🎯 Models are ready for deployment!")

def run_web_app():
    """Run the Flask web application"""
    print("🌐 Starting Flask web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Import and run Flask app
    from app.app import app, initialize_app
    
    # Initialize the app first
    if initialize_app():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start web application. Please check the model files.")

def make_prediction():
    """Make a prediction from command line"""
    print("🔮 Car Price Prediction (CLI Mode)")
    print("Enter car specifications (press Enter to use defaults):")
    
    # Sample car specifications
    sample_car = {
        'symboling': 0,
        'fueltype': 0,  # gas
        'enginesize': 130,
        'horsepower': 111,
        'peakrpm': 5000,
        'wheelbase': 88.6,
        'carlength': 168.8,
        'carwidth': 64.1,
        'carheight': 48.8,
        'curbweight': 2548,
        'citympg': 21,
        'highwaympg': 27
    }
    
    # Load model
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
            
    except FileNotFoundError:
        print("❌ Model files not found. Please run 'python main.py train' first.")
        return
    
    # Get user input
    car_specs = {}
    for feature in feature_columns:
        default = sample_car.get(feature, 0)
        user_input = input(f"{feature} (default: {default}): ").strip()
        car_specs[feature] = float(user_input) if user_input else default
    
    # Make prediction
    features = [car_specs[col] for col in feature_columns]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    
    print(f"\n🎯 Predicted Car Price: ${prediction:,.2f}")

def generate_report():
    """Generate analysis report"""
    print("📊 Generating Analysis Report...")
    
    # Load data
    df = load_and_clean_data('data/CarPrice_Assignment.csv')
    
    # Basic statistics
    print("\n📈 Dataset Overview:")
    print(f"   • Total cars: {len(df)}")
    print(f"   • Features: {len(df.columns) - 1}")
    print(f"   • Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"   • Average price: ${df['price'].mean():,.0f}")
    
    # Feature analysis
    print("\n🔍 Key Features Analysis:")
    print(f"   • Engine size range: {df['enginesize'].min()} - {df['enginesize'].max()}")
    print(f"   • Horsepower range: {df['horsepower'].min()} - {df['horsepower'].max()}")
    print(f"   • Fuel types: {df['fueltype'].unique()}")
    print(f"   • Body styles: {df['carbody'].unique()}")
    
    # Correlation with price
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numerical_cols].corr()['price'].sort_values(ascending=False)
    
    print("\n💰 Top Price Correlations:")
    for feature, corr in correlations[1:6].items():
        print(f"   • {feature}: {corr:.3f}")
    
    print("\n✅ Report generated successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Car Price Prediction Project')
    parser.add_argument('command', choices=['train', 'web', 'predict', 'report'],
                       help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_models()
    elif args.command == 'web':
        run_web_app()
    elif args.command == 'predict':
        make_prediction()
    elif args.command == 'report':
        generate_report()

if __name__ == '__main__':
    main() 