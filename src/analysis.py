import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import data loading functions
from data_loader import load_all_data
from data_collection import collect_economic_data, collect_stock_data

def explore_airline_data(airline_data):
    """Explore airline quarterly metrics"""
    print("=== AIRLINE DATA ANALYSIS ===")
    print(f"Shape: {airline_data.shape}")
    print(f"Airlines: {airline_data['ticker'].unique()}")
    print(f"Time period: {airline_data['quarter'].unique()}")
    print(f"Years: {airline_data['year'].unique()}")
    
    # Check for missing values
    print("\nMissing values:")
    print(airline_data.isnull().sum())
    
    # Basic statistics
    print("\nKey metrics summary:")
    numeric_cols = ['load_factor', 'prasm', 'casm', 'fuel_cost_per_gallon']
    if all(col in airline_data.columns for col in numeric_cols):
        print(airline_data[numeric_cols].describe())
    
    # Plot load factors by airline
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if 'load_factor' in airline_data.columns:
        for airline in airline_data['ticker'].unique():
            data_subset = airline_data[airline_data['ticker'] == airline]
            quarters = [f"{row['quarter']} {row['year']}" for _, row in data_subset.iterrows()]
            plt.plot(quarters, data_subset['load_factor'], marker='o', label=airline)
    
    plt.title('Load Factor by Quarter')
    plt.xlabel('Quarter')
    plt.ylabel('Load Factor (%)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot PRASM vs CASM
    plt.subplot(1, 2, 2)
    if 'prasm' in airline_data.columns and 'casm' in airline_data.columns:
        for airline in airline_data['ticker'].unique():
            data_subset = airline_data[airline_data['ticker'] == airline]
            plt.scatter(data_subset['casm'], data_subset['prasm'], label=airline, s=50, alpha=0.7)
    
    plt.title('PRASM vs CASM')
    plt.xlabel('CASM (Cost per ASM)')
    plt.ylabel('PRASM (Revenue per ASM)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/airline_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def explore_tsa_data(tsa_data):
    """Explore TSA throughput patterns"""
    print("\n=== TSA DATA ANALYSIS ===")
    print(f"Shape: {tsa_data.shape}")
    print(f"Date range: {tsa_data['Date'].min()} to {tsa_data['Date'].max()}")
    
    # Convert date column
    tsa_data['Date'] = pd.to_datetime(tsa_data['Date'])
    
    # Daily passenger statistics - use 'Totals' column
    passenger_col = 'Totals'  # Your actual column name

    print(f"\nHourly passenger statistics:")
    print(tsa_data[passenger_col].describe())

    # Create daily aggregates first (sum hourly data by date)
    daily_passengers = tsa_data.groupby('Date')[passenger_col].sum().reset_index()

    # Create monthly aggregates for visualization
    daily_passengers['year_month'] = daily_passengers['Date'].dt.to_period('M')
    monthly_passengers = daily_passengers.groupby('year_month')[passenger_col].sum().reset_index()
    monthly_passengers['year_month'] = monthly_passengers['year_month'].astype(str)

    # Plot monthly trends
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(monthly_passengers['year_month'], monthly_passengers[passenger_col], marker='o')
    plt.title('Monthly TSA Passenger Throughput')
    plt.xlabel('Month')
    plt.ylabel('Total Passengers')
    plt.xticks(rotation=45)

    # Day of week patterns (use daily aggregates)
    plt.subplot(1, 2, 2)
    daily_passengers['day_of_week'] = daily_passengers['Date'].dt.day_name()
    daily_avg = daily_passengers.groupby('day_of_week')[passenger_col].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg_ordered = daily_avg.reindex(day_order)

    plt.bar(daily_avg_ordered.index, daily_avg_ordered.values)
    plt.title('Average Daily Passengers by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Daily Passengers')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('results/tsa_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlations(airline_data, tsa_data):
    """Analyze relationships between datasets"""
    print("\n=== CORRELATION ANALYSIS ===")
    
    try:
        # Convert dates and create quarterly aggregates of TSA data
        tsa_data['Date'] = pd.to_datetime(tsa_data['Date'])
        
        # Find passenger column
        passenger_col = 'Totals'  
        
        if passenger_col:
            # Create quarterly TSA aggregates
            tsa_data['quarter'] = tsa_data['Date'].dt.quarter
            tsa_data['year'] = tsa_data['Date'].dt.year
            
            quarterly_tsa = tsa_data.groupby(['year', 'quarter'])[passenger_col].sum().reset_index()
            quarterly_tsa['quarter'] = 'Q' + quarterly_tsa['quarter'].astype(str)
            
            # Merge with airline data
            merged_data = pd.merge(
                airline_data, 
                quarterly_tsa, 
                on=['year', 'quarter'], 
                how='inner'
            )
            
            if len(merged_data) > 0:
                print(f"Successfully merged {len(merged_data)} records")
                
                # Calculate correlations
                numeric_cols = ['load_factor', 'prasm', 'casm', passenger_col]
                available_cols = [col for col in numeric_cols if col in merged_data.columns]
                
                if len(available_cols) > 1:
                    correlation_matrix = merged_data[available_cols].corr()
                    
                    print("\nCorrelation Matrix:")
                    print(correlation_matrix.round(3))
                    
                    # Plot correlation heatmap
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                    plt.title('Correlation Matrix: TSA vs Airline Metrics')
                    plt.tight_layout()
                    plt.savefig('results/correlation_analysis.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    # Scatter plot: TSA vs Load Factor
                    if 'load_factor' in merged_data.columns:
                        plt.figure(figsize=(8, 6))
                        for airline in merged_data['ticker'].unique():
                            subset = merged_data[merged_data['ticker'] == airline]
                            plt.scatter(subset[passenger_col], subset['load_factor'], 
                                      label=airline, alpha=0.7, s=60)
                        
                        plt.xlabel('Quarterly TSA Passengers')
                        plt.ylabel('Load Factor (%)')
                        plt.title('TSA Throughput vs Airline Load Factor')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig('results/tsa_vs_load_factor.png', dpi=300, bbox_inches='tight')
                        plt.show()
            else:
                print("Warning: No overlapping data between TSA and airline datasets")
        else:
            print("Warning: Could not find passenger column in TSA data")
            
    except Exception as e:
        print(f"Error in correlation analysis: {e}")

def explore_economic_data():
    """Explore economic indicators"""
    print("\n=== ECONOMIC DATA ANALYSIS ===")
    
    try:
        fuel_data, confidence_data, gdp_data = collect_economic_data()
        
        print("Economic indicators loaded successfully")
        
        # Plot economic trends
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Fuel prices
        axes[0].plot(fuel_data.index, fuel_data.values)
        axes[0].set_title('Jet Fuel Prices')
        axes[0].set_ylabel('Price per Gallon ($)')
        
        # Consumer confidence
        axes[1].plot(confidence_data.index, confidence_data.values)
        axes[1].set_title('Consumer Confidence')
        axes[1].set_ylabel('Confidence Index')
        
        # GDP
        axes[2].plot(gdp_data.index, gdp_data.values)
        axes[2].set_title('GDP')
        axes[2].set_ylabel('GDP (Billions)')
        axes[2].set_xlabel('Date')
        
        plt.tight_layout()
        plt.savefig('results/economic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fuel_data, confidence_data, gdp_data
        
    except Exception as e:
        print(f"Could not load economic data: {e}")
        return None, None, None

def main():
    """Run complete analysis"""
    print("Starting comprehensive data analysis...")
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Load manual data
    try:
        data = load_all_data()
        airline_data = data['airlines']
        tsa_data = data['tsa']
        
        # Run analyses
        explore_airline_data(airline_data)
        explore_tsa_data(tsa_data)
        analyze_correlations(airline_data, tsa_data)
        explore_economic_data()
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Key findings:")
        print("- Check correlation matrix for TSA vs airline metrics")
        print("- Look for seasonal patterns in both datasets")
        print("- Review data quality and missing values")
        print("- All plots saved to results/ folder")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Make sure your data files are in the correct location")

if __name__ == "__main__":
    main()