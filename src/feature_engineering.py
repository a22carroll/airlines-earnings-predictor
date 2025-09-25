import pandas as pd
import numpy as np
from data_loader import load_all_data
from data_collection import collect_economic_data

def create_tsa_quarterly_features(tsa_data):
    """Convert daily TSA data to quarterly features"""
    print("Creating TSA quarterly features...")
    
    tsa_data = tsa_data.copy()
    tsa_data['Date'] = pd.to_datetime(tsa_data['Date'])
    
    # Extract quarter and year
    tsa_data['quarter'] = tsa_data['Date'].dt.quarter
    tsa_data['year'] = tsa_data['Date'].dt.year
    tsa_data['quarter'] = 'Q' + tsa_data['quarter'].astype(str)
    
    # Aggregate by quarter
    quarterly_tsa = tsa_data.groupby(['year', 'quarter']).agg({
        'Totals': ['sum', 'mean', 'std']
    }).round(0)
    
    # Flatten column names
    quarterly_tsa.columns = ['tsa_total', 'tsa_daily_avg', 'tsa_daily_std']
    quarterly_tsa = quarterly_tsa.reset_index()
    
    # Create growth rates
    quarterly_tsa = quarterly_tsa.sort_values(['year', 'quarter'])
    quarterly_tsa['tsa_growth'] = quarterly_tsa['tsa_total'].pct_change() * 100
    
    # Quarter number for seasonality
    quarter_mapping = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    quarterly_tsa['quarter_num'] = quarterly_tsa['quarter'].map(quarter_mapping)
    
    print(f"Created TSA features for {len(quarterly_tsa)} quarters")
    return quarterly_tsa

def create_airline_features(airline_data):
    """Create features from airline data"""
    print("Creating airline features...")
    
    features = airline_data.copy()
    
    # Profit margin
    if 'prasm' in features.columns and 'casm' in features.columns:
        features['profit_margin'] = features['prasm'] - features['casm']
    
    # Sort for lag features
    features = features.sort_values(['ticker', 'year', 'quarter'])
    
    # Previous quarter load factor
    features['load_factor_lag1'] = features.groupby('ticker')['load_factor'].shift(1)
    features['load_factor_change'] = features['load_factor'] - features['load_factor_lag1']
    
    # Industry comparison (vs average in same quarter)
    features['load_factor_vs_industry'] = features['load_factor'] - features.groupby(['year', 'quarter'])['load_factor'].transform('mean')
    
    print(f"Created airline features for {len(features)} records")
    return features

def create_economic_features():
    """Create quarterly economic features"""
    print("Creating economic features...")
    
    try:
        fuel_data, confidence_data, gdp_data = collect_economic_data()
        
        economic_features = pd.DataFrame()
        
        # Process fuel prices (daily to quarterly)
        if fuel_data is not None and not fuel_data.empty:
            fuel_df = fuel_data.reset_index()
            fuel_df.columns = ['date', 'oil_price']
            fuel_df['date'] = pd.to_datetime(fuel_df['date'])
            fuel_df['quarter'] = fuel_df['date'].dt.quarter
            fuel_df['year'] = fuel_df['date'].dt.year
            fuel_df['quarter'] = 'Q' + fuel_df['quarter'].astype(str)
            
            fuel_quarterly = fuel_df.groupby(['year', 'quarter']).agg({
                'oil_price': ['mean', 'std']
            }).round(2)
            
            fuel_quarterly.columns = ['oil_price_avg', 'oil_price_volatility']
            fuel_quarterly = fuel_quarterly.reset_index()
            fuel_quarterly['oil_price_change'] = fuel_quarterly['oil_price_avg'].pct_change() * 100
            
            economic_features = fuel_quarterly
        
        # Process consumer confidence (monthly to quarterly) 
        if confidence_data is not None and not confidence_data.empty:
            conf_df = confidence_data.reset_index()
            conf_df.columns = ['date', 'consumer_confidence']
            conf_df['date'] = pd.to_datetime(conf_df['date'])
            conf_df['quarter'] = conf_df['date'].dt.quarter
            conf_df['year'] = conf_df['date'].dt.year
            conf_df['quarter'] = 'Q' + conf_df['quarter'].astype(str)
            
            conf_quarterly = conf_df.groupby(['year', 'quarter']).agg({
                'consumer_confidence': 'mean'
            }).round(1)
            
            conf_quarterly.columns = ['consumer_confidence']
            conf_quarterly = conf_quarterly.reset_index()
            conf_quarterly['confidence_change'] = conf_quarterly['consumer_confidence'].pct_change() * 100
            
            # Merge with economic features
            if economic_features.empty:
                economic_features = conf_quarterly
            else:
                economic_features = pd.merge(economic_features, conf_quarterly, on=['year', 'quarter'], how='outer')
        
        # Process GDP (already quarterly)
        if gdp_data is not None and not gdp_data.empty:
            gdp_df = gdp_data.reset_index()
            gdp_df.columns = ['date', 'gdp']
            gdp_df['date'] = pd.to_datetime(gdp_df['date'])
            gdp_df['quarter'] = gdp_df['date'].dt.quarter
            gdp_df['year'] = gdp_df['date'].dt.year
            gdp_df['quarter'] = 'Q' + gdp_df['quarter'].astype(str)
            
            gdp_df['gdp_growth'] = gdp_df['gdp'].pct_change(4) * 100  # Year-over-year
            gdp_quarterly = gdp_df[['year', 'quarter', 'gdp', 'gdp_growth']]
            
            # Merge with economic features
            if economic_features.empty:
                economic_features = gdp_quarterly
            else:
                economic_features = pd.merge(economic_features, gdp_quarterly, on=['year', 'quarter'], how='outer')
        
        print(f"Created economic features for {len(economic_features)} quarters")
        return economic_features
        
    except Exception as e:
        print(f"Error creating economic features: {e}")
        return pd.DataFrame()

def merge_all_features():
    """Combine all feature sets into final dataset"""
    print("Merging all features...")
    
    # Load base data
    data = load_all_data()
    airline_data = data['airlines']
    tsa_data = data['tsa']
    
    # Create feature sets
    tsa_features = create_tsa_quarterly_features(tsa_data)
    airline_features = create_airline_features(airline_data)
    economic_features = create_economic_features()
    
    # Start with airline features as base
    final_dataset = airline_features.copy()
    
    # Merge TSA features
    if not tsa_features.empty:
        final_dataset = pd.merge(
            final_dataset, 
            tsa_features, 
            on=['year', 'quarter'], 
            how='left'
        )
        print(f"Merged TSA features: {len(final_dataset)} records")
    
    # Merge economic features
    if not economic_features.empty:
        final_dataset = pd.merge(
            final_dataset, 
            economic_features, 
            on=['year', 'quarter'], 
            how='left'
        )
        print(f"Merged economic features: {len(final_dataset)} records")
    
    # Handle missing values (forward fill for economic data gaps)
    numeric_columns = final_dataset.select_dtypes(include=[np.number]).columns
    final_dataset[numeric_columns] = final_dataset[numeric_columns].fillna(method='ffill')
    
    print(f"Final dataset shape: {final_dataset.shape}")
    print(f"Features created: {list(final_dataset.columns)}")
    
    return final_dataset

def save_features(dataset, filename='modeling_dataset.csv'):
    """Save the final feature dataset"""
    filepath = f'data/processed/{filename}'
    dataset.to_csv(filepath, index=False)
    print(f"Features saved to: {filepath}")
    return filepath

def main():
    """Create and save all features"""
    print("Starting feature engineering...")
    
    # Create results directory
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Create final dataset
        final_features = merge_all_features()
        
        # Save to file
        save_features(final_features)
        
        # Show summary
        print("\n=== FEATURE ENGINEERING COMPLETE ===")
        print(f"Dataset shape: {final_features.shape}")
        print(f"Airlines: {final_features['ticker'].nunique()}")
        print(f"Quarters: {len(final_features.groupby(['year', 'quarter']))}")
        
        # Show feature columns
        feature_cols = [col for col in final_features.columns 
                       if col not in ['ticker', 'quarter', 'year']]
        print(f"Features created: {len(feature_cols)}")
        
        # Check for missing values
        missing_summary = final_features.isnull().sum()
        if missing_summary.sum() > 0:
            print("\nMissing values:")
            print(missing_summary[missing_summary > 0])
        else:
            print("No missing values - dataset ready for modeling!")
            
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()