# 🚗 Car Price Prediction Project

A comprehensive machine learning project that predicts car prices using multiple traditional ML algorithms. This project demonstrates end-to-end data science workflow from exploratory data analysis to model deployment.

## 📊 Project Overview

This project analyzes a dataset of 205 cars with various features (engine specifications, dimensions, fuel efficiency, etc.) to predict car prices. The goal is to build a reliable model that can estimate car prices based on their characteristics.

### Key Features:
- **4 Machine Learning Models**: Linear Regression, Decision Tree, Random Forest, XGBoost
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Engineering**: Proper preprocessing and encoding of categorical variables
- **Model Comparison**: Systematic evaluation using multiple metrics (RMSE, MAE, R²)
- **Modular Code**: Reusable Python scripts for preprocessing and model training
- **Web Application**: Interactive Flask web app for real-time predictions
- **Model Persistence**: Save and load trained models for deployment
- **Professional Structure**: Complete project organization with proper documentation

## 🏗️ Project Structure

```
car-price-prediction/
│
├── data/
│   └── CarPrice_Assignment.csv          # Original dataset
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb   # Exploratory data analysis
│   └── 02_Modeling_and_Evaluation.ipynb # Model training and evaluation
│
├── src/
│   ├── data_preprocessing.py            # Data cleaning and preprocessing functions
│   └── model_training.py                # Model training and evaluation class
│
├── app/
│   ├── app.py                          # Flask web application
│   └── templates/
│       └── index.html                  # Web interface template
│
├── models/                             # Saved trained models (auto-generated)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── main.py                             # Main execution script
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore file
└── README.md                          # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Code-with-Qaisan/Car-Price-Prediction.git
   cd Car-Price-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models and run web app**
   ```bash
   # Train models
   python main.py train
   
   # Run web application
   python main.py web
   
   # Or run notebooks
   jupyter notebook notebooks/
   ```

## 📈 Data Analysis Highlights

### Dataset Characteristics
- **205 samples** with 26 features
- **Target variable**: Car price
- **Features include**: Engine specifications, dimensions, fuel efficiency, safety ratings

### Key Insights from EDA
- Price distribution shows right-skewed pattern
- Strong correlations between engine size, horsepower, and price
- Categorical features like fuel type and body style significantly impact pricing
- No missing values in the dataset

## 🤖 Models Implemented

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | ~3,200 | ~2,400 | ~0.85 |
| Decision Tree | ~2,800 | ~2,100 | ~0.88 |
| Random Forest | ~2,400 | ~1,800 | ~0.92 |
| XGBoost | ~2,200 | ~1,700 | ~0.93 |

**Best Performing Model**: XGBoost with R² = 0.93

## 🔧 Technical Implementation

### Data Preprocessing
- **Missing Value Handling**: No missing values found
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Data Cleaning**: Standardized car brand names

### Model Training
- **Train-Test Split**: 80-20 split with random_state=42
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Hyperparameter Tuning**: Default parameters with random_state for reproducibility

### Evaluation Metrics
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: Coefficient of determination (higher is better)

## 📊 Key Findings

1. **Ensemble Methods Excel**: Random Forest and XGBoost significantly outperform simpler models
2. **Feature Importance**: Engine size, horsepower, and curb weight are top predictors
3. **Model Interpretability**: Decision Tree provides good balance between performance and interpretability

## 🛠️ Usage

### Using the Python Scripts

```python
from src.data_preprocessing import load_and_clean_data, prepare_features
from src.model_training import CarPricePredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = load_and_clean_data('data/CarPrice_Assignment.csv')
X, y = prepare_features(df)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
predictor = CarPricePredictor()
results = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)

# Get best model
best_model_name, best_model_results = predictor.get_best_model()
print(f"Best model: {best_model_name}")
```

## 🌐 Web Application

The project includes a **Flask web application** that provides an interactive interface for car price prediction:

### Features:
- **User-friendly interface** with Bootstrap styling
- **Real-time predictions** based on car specifications
- **Responsive design** that works on desktop and mobile
- **Input validation** and error handling

### How to Use:
1. Train the models: `python main.py train`
2. Start the web app: `python main.py web`
3. Open browser to: `http://localhost:5000`
4. Enter car specifications and get instant price predictions

## 📝 Future Enhancements

- [x] Web application using Flask ✅
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Model deployment with API
- [ ] Real-time price prediction interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 👨‍💻 Author

Qaisan Raza - qaisanraza0@gmail.com

---

**Note**: This project is designed for educational and portfolio purposes. The models demonstrate machine learning concepts and should not be used for actual car price predictions without proper validation and domain expertise. 
