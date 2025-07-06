from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from src.data_preprocessing import load_and_clean_data, prepare_features
from src.model_training import CarPricePredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store model and scaler
model = None
scaler = None
feature_columns = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler, feature_columns
    
    try:
        # Check if model files exist
        if not os.path.exists('models/best_model.pkl') or \
           not os.path.exists('models/scaler.pkl') or \
           not os.path.exists('models/feature_columns.pkl'):
            print("Model files not found. Training new model...")
            return False
        
        # Load the best model
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature columns
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Verify that all components are loaded
        if model is None or scaler is None or feature_columns is None:
            print("Error: Some model components are None. Training new model...")
            return False
            
        print("Model loaded successfully!")
        print(f"Feature columns: {len(feature_columns)} features")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        return False

def train_and_save_model():
    """Train the model and save it"""
    global model, scaler, feature_columns
    
    # Load and preprocess data
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
    predictor = CarPricePredictor()
    results = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Get best model
    best_model_name, best_model = predictor.get_best_model()
    model = best_model
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print(f"Model saved successfully! Best model: {best_model_name}")

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict car price"""
    try:
        # Check if model and feature columns are loaded
        if model is None or scaler is None or feature_columns is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the application.'
            })
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            })
        
        # Create feature vector
        features = []
        for col in feature_columns:
            if col in data:
                features.append(float(data[col]))
            else:
                features.append(0.0)  # Default value
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/features')
def get_features():
    """Get available features for the form"""
    if feature_columns is None:
        return jsonify({
            'error': 'Feature columns not loaded'
        })
    return jsonify({
        'features': feature_columns
    })

@app.route('/api/status')
def get_status():
    """Get model status"""
    return jsonify({
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_columns_loaded': feature_columns is not None,
        'feature_count': len(feature_columns) if feature_columns else 0
    })

def initialize_app():
    """Initialize the Flask app with model loading"""
    global model, scaler, feature_columns
    
    print("🔧 Initializing Flask application...")
    
    # Load or train model on startup
    if not load_model():
        print("🔄 Training new model...")
        train_and_save_model()
    
    # Verify model is loaded
    if model is None or scaler is None or feature_columns is None:
        print("❌ Failed to load model. Please check the model files.")
        return False
    
    print("✅ Flask application initialized successfully!")
    return True

if __name__ == '__main__':
    # Initialize the app
    if initialize_app():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize application. Exiting...") 