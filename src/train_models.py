import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_modeling_data():
    """Load the feature-engineered dataset for training"""
    print("Loading modeling dataset...")
    
    # Load the dataset created by feature engineering
    dataset = pd.read_csv('data/processed/modeling_dataset.csv')
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Airlines: {dataset['ticker'].unique()}")
    print(f"Date range: Q{dataset['quarter_num'].min()} {dataset['year'].min()} to Q{dataset['quarter_num'].max()} {dataset['year'].max()}")
    
    return dataset

def prepare_features_and_target(dataset, target_column='total_revenue_millions'):
    """
    Separate features from target variable and handle missing values
    
    Args:
        dataset: The complete dataset with features and target
        target_column: Name of column to predict (default: total_revenue_millions)
    
    Returns:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature column names
    """
    print(f"Preparing features to predict: {target_column}")
    
    # Select only top features to prevent overfitting
    selected_features = [
    'passenger_revenue_millions', 
    'total_operating_expenses_millions',
    'load_factor', 
    'tsa_total', 
    'oil_price_avg'
]

# Extract features and target
    X = dataset[selected_features].copy()
    y = dataset[target_column].copy()
    
    # Handle missing values in features
    # Forward fill for time series data, then backward fill any remaining
    X = X.ffill().bfill()
    
    # Remove any rows where target is missing
    valid_rows = ~y.isna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Selected features: {selected_features}")
    
    return X, y, selected_features

def create_time_aware_split(dataset, test_size=0.25):
    """
    Create train/test split that respects time series nature
    Uses most recent quarters for testing to simulate real prediction
    
    Args:
        dataset: Complete dataset with year/quarter columns
        test_size: Proportion of data to use for testing
    
    Returns:
        train_idx, test_idx: Indices for train and test sets
    """
    print("Creating time-aware train/test split...")
    
    # Sort by time
    dataset_sorted = dataset.sort_values(['year', 'quarter_num']).reset_index(drop=True)
    
    # Calculate split point (use most recent data for testing)
    n_samples = len(dataset_sorted)
    n_test = int(n_samples * test_size)
    split_point = n_samples - n_test
    
    train_idx = dataset_sorted.index[:split_point]
    test_idx = dataset_sorted.index[split_point:]
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples: {len(test_idx)}")
    
    return train_idx, test_idx

def train_baseline_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Train multiple baseline models and compare their performance
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data  
        feature_names: Names of features for interpretability
    
    Returns:
        models: Dictionary of trained models
        results: Dictionary of performance metrics
    """
    print("Training baseline models...")
    
    # Initialize models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Store results
    results = {}
    trained_models = {}
    
    # Scale features for linear models (helps with numerical stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled features for linear models, original for tree-based
        if 'Linear' in name or 'Ridge' in name:
            model.fit(X_train_scaled, y_train)
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
        
        # Calculate performance metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Store results
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': test_pred
        }
        
        trained_models[name] = {'model': model, 'scaler': scaler if 'Linear' in name or 'Ridge' in name else None}
        
        print(f"  Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
        print(f"  Train MAE: {train_mae:.1f}, Test MAE: {test_mae:.1f}")
    
    return trained_models, results

def analyze_feature_importance(model, feature_names, model_name):
    """
    Extract and display feature importance from trained model
    
    Args:
        model: Trained model with feature importance
        feature_names: Names of features
        model_name: Name of model for plotting
    """
    print(f"\nAnalyzing feature importance for {model_name}...")
    
    # Extract feature importance (different methods for different models)
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models - use absolute values of coefficients
        importance = np.abs(model.coef_)
    else:
        print(f"Cannot extract feature importance for {model_name}")
        return
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    # Keep only top 5-8 features based on business logic
    top_features = feature_importance.head(5)    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Feature Importance - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower().replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

def plot_model_comparison(results):
    """
    Create visualizations comparing model performance
    
    Args:
        results: Dictionary containing model results
    """
    print("Creating model comparison plots...")
    
    # Prepare data for plotting
    model_names = list(results.keys())
    test_r2 = [results[name]['test_r2'] for name in model_names]
    test_mae = [results[name]['test_mae'] for name in model_names]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² comparison
    bars1 = ax1.bar(model_names, test_r2, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance - R² Score (Higher is Better)')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars1, test_r2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # MAE comparison  
    bars2 = ax2.bar(model_names, test_mae, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Model Performance - MAE (Lower is Better)')
    
    # Add value labels on bars
    for bar, value in zip(bars2, test_mae):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions_vs_actual(y_test, results, dataset_test):
    """
    Plot predicted vs actual values for each model
    
    Args:
        y_test: Actual target values
        results: Model results containing predictions
        dataset_test: Test dataset for additional context
    """
    print("Creating prediction vs actual plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        predictions = result['predictions']
        
        # Scatter plot of predicted vs actual
        ax.scatter(y_test, predictions, alpha=0.7)
        
        # Perfect prediction line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Revenue (Millions)')
        ax.set_ylabel('Predicted Revenue (Millions)')
        ax.set_title(f'{model_name}\nR² = {result["test_r2"]:.3f}')
        ax.legend()
        
        # Add grid for easier reading
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

def cross_validate_best_model(X, y, best_model):
    """
    Perform cross-validation on the best performing model
    
    Args:
        X: Feature matrix
        y: Target variable
        best_model: The best performing model
    
    Returns:
        cv_scores: Cross-validation scores
    """
    print("Performing cross-validation on best model...")
    
    # Use TimeSeriesSplit for proper time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Perform cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=tscv, scoring='r2')
    
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return cv_scores

def save_trained_models(models, results, filename='trained_models.pkl'):
    """
    Save trained models and results to disk
    
    Args:
        models: Dictionary of trained models
        results: Dictionary of model results
        filename: Name of file to save models
    """
    import pickle
    import os
    
    os.makedirs('models', exist_ok=True)
    
    # Save models and results
    model_data = {
        'models': models,
        'results': results
    }
    
    filepath = f'models/{filename}'
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Models saved to: {filepath}")

def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    try:
        # 1. Load the feature-engineered dataset
        dataset = load_modeling_data()
        
        # 2. Prepare features and target variable
        X, y, feature_names = prepare_features_and_target(dataset)
        
        # 3. Create time-aware train/test split
        train_idx, test_idx = create_time_aware_split(dataset)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 4. Train baseline models
        models, results = train_baseline_models(X_train, y_train, X_test, y_test, feature_names)
        
        # 5. Find best performing model
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        best_model = models[best_model_name]['model']
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Test R²: {results[best_model_name]['test_r2']:.3f}")
        
        # 6. Analyze feature importance for best model
        analyze_feature_importance(best_model, feature_names, best_model_name)
        
        # 7. Create performance visualizations
        plot_model_comparison(results)
        plot_predictions_vs_actual(y_test, results, dataset.iloc[test_idx])
        
        # 8. Cross-validate best model
        cross_validate_best_model(X, y, best_model)
        
        # 9. Save trained models
        save_trained_models(models, results)
        
        # 10. Summary report
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETE")
        print("="*50)
        print(f"Best Model: {best_model_name}")
        print(f"Test R² Score: {results[best_model_name]['test_r2']:.3f}")
        print(f"Test MAE: {results[best_model_name]['test_mae']:.1f} million")
        print("\nKey insights:")
        print("- Models trained on airline operational and TSA passenger data")
        print("- Time series split used to simulate real-world prediction scenario")
        print("- Feature importance analysis shows which factors drive earnings")
        print("\nFiles created:")
        print("- results/model_comparison.png")
        print("- results/predictions_vs_actual.png")
        print("- results/feature_importance.png")
        print("- models/trained_models.pkl")
        
    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()